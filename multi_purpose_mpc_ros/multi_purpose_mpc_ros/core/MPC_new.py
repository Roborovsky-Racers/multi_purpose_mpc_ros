from typing import Tuple
import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

# Colors
PREDICTION = '#BA4A00'

##################
# MPC Controller #
##################


class MPCNew:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, use_obstacle_avoidance, use_path_constraints_topic,
                 D_s=1, D_a=1):
        """
        Constructor for the Model Predictive Controller.
        :param model: bicycle model object to be controlled
        :param N: time horizon | int
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix
        :param StateConstraints: dictionary of state constraints
        :param InputConstraints: dictionary of input constraints
        :param ay_max: maximum allowed lateral acceleration in curves
        """
        # Parameters
        self.N = N  # horizon
        self.Q = Q  # weight matrix state vector
        self.R = R  # weight matrix input vector
        self.QN = QN  # weight matrix terminal
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic

        # Model
        self.model = model

        # Delays
        self.D_s = D_s  # Steering delay steps
        self.D_a = D_a  # Acceleration delay steps

        # Dimensions
        self.nx = self.model.n_states + self.D_s + self.D_a  # Augmented state dimension
        self.nu = 2  # Number of inputs

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Maximum lateral acceleration
        self.ay_max = ay_max

        # Current control and prediction
        self.current_prediction = None

        # Counter for old control signals in case of infeasible problem
        self.infeasibility_counter = 0

        # Current control signals
        self.current_control = np.zeros((self.nu * self.N))

        # Initialize delayed input buffers
        self.delayed_steering = [0.0 for _ in range(self.D_s)]
        self.delayed_acceleration = [0.0 for _ in range(self.D_a)]

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

        # Initialize dynamic constraints once if obstacle_avoidance is disabled
        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N,
                self.model.safety_margin)

    def _init_problem(self, N, safety_margin, force_update_dynamic_constraints=False):
        """
        Initialize optimization problem for current time step.
        """
        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']
        print(f"umin: {umin}, umax: {umax}, xmin: {xmin}, xmax: {xmax}")

        # Precompute common terms
        nx = self.nx
        nu = self.nu
        nx_N = nx * (N + 1)
        nu_N = nu * N

        # LTV System Matrices
        A = np.zeros((nx_N, nx_N))
        B = np.zeros((nx_N, nu_N))

        # Reference vector for state and input variables
        ur = np.zeros(nu_N)
        xr = np.zeros(nx_N)
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(N * nx)

        # Dynamic constraints
        xmin_dyn = np.kron(np.ones(N + 1), np.concatenate([xmin, [-np.inf] * (self.D_s + self.D_a)]))
        xmax_dyn = np.kron(np.ones(N + 1), np.concatenate([xmax, [np.inf] * (self.D_s + self.D_a)]))
        umax_dyn = np.kron(np.ones(N), umax)

        # Iterate over horizon
        for n in range(N):
            # Get information about current waypoint
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            v_ref = current_waypoint.v_ref

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s, self.D_s, self.D_a)

            idx_x = (n + 1) * nx
            idx_x_prev = n * nx
            idx_u = n * nu

            A[idx_x: idx_x + nx, idx_x_prev: idx_x_prev + nx] = A_lin
            B[idx_x: idx_x + nx, idx_u: idx_u + nu] = B_lin

            # Set reference for input signal
            ur[idx_u:idx_u + nu] = [v_ref, kappa_ref]
            # Compute equality constraint offset (B*ur)
            uq[n * nx:(n + 1) * nx] = B_lin.dot([v_ref, kappa_ref]) - f

            # Constrain maximum speed based on curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_ref) + 1e-12))
            umax_dyn[nu * n] = min(vmax_dyn, umax_dyn[nu * n])

        # Update path constraints
        if force_update_dynamic_constraints:
            ub, lb = self.model.reference_path.update_simple_path_constraints_horizon(
                self.model.wp_id + 1,
                N,
                safety_margin)
            self.model.reference_path.border_cells.current_wp_id = self.model.wp_id

        elif (self.use_obstacle_avoidance and not self.use_path_constraints_topic):
            ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id + 1,
                [self.model.temporal_state.x, self.model.temporal_state.y, self.model.temporal_state.psi],
                N,
                self.model.length,
                self.model.width,
                safety_margin)
        else:
            ub = self.model.reference_path.path_constraints[0][self.model.wp_id]
            lb = self.model.reference_path.path_constraints[1][self.model.wp_id]
            self.model.reference_path.border_cells.current_wp_id = self.model.wp_id

        # Update dynamic state constraints
        xmin_dyn[0:self.nx] = xmax_dyn[0:self.nx] = np.concatenate([self.model.spatial_state[:], self.delayed_steering + self.delayed_acceleration])
        xmin_dyn[nx::nx] = lb
        xmax_dyn[nx::nx] = ub

        # Set reference for state as center-line of drivable area
        xr[nx::nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(N + 1),
                         -sparse.eye(nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye(nx_N + nu_N)
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Construct bounds
        x0 = np.concatenate([self.model.spatial_state[:], self.delayed_steering + self.delayed_acceleration])
        # print(f"x0: {x0}, uq: {uq}")
        # print(f"spatial_state: {self.model.spatial_state[:]}\n, deleayed_steering: {self.delayed_steering}\n, delayed_acceleration: {self.delayed_acceleration}")
        leq = np.hstack([-x0, uq])
        ueq = leq
        lineq = np.hstack([xmin_dyn, np.kron(np.ones(N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        Q_aug = sparse.block_diag([self.Q, sparse.eye(self.D_s + self.D_a) * 0.0], format='csc')
        QN_aug = Q_aug  # 終端コストも同様に拡張

        # P 行列の設定
        P = sparse.block_diag([
            sparse.kron(sparse.eye(N + 1), Q_aug),    # 状態コスト
            sparse.kron(sparse.eye(N), self.R)        # 入力コスト
        ], format='csc')

        # 変数の総数 n を計算
        n = (N + 1) * nx + N * nu

        # q ベクトルの設定
        q = np.hstack([
            -np.tile(np.diag(Q_aug.toarray()), N + 1) * xr,
            -np.tile(np.diag(self.R.toarray()), N) * ur
        ])

        print(f"P.shape: {P.shape}, q.shape: {q.shape}, A.shape: {A.shape}, l.shape: {l.shape}, u.shape: {u.shape}")

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self) -> Tuple[np.ndarray, float]:
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # Number of state variables
        nx = self.nx
        nu = self.nu

        # Update current waypoint
        self.model.get_current_waypoint()

        # check if the horizon exceeds the number of waypoints
        N = self.N
        if self.model.wp_id >= self.model.reference_path.n_waypoints and not self.model.reference_path.circular:
            N = self.model.reference_path.n_waypoints - self.model.wp_id
            if N <= 0:
                print("End of path reached. Stopping the vehicle.")
                u = np.array([0.0, 0.0])
                max_delta = 0.0
                return u, max_delta


        # Update spatial state
        self.model.spatial_state = self.model.t2s(reference_waypoint=
            self.model.current_waypoint, reference_state=
            self.model.temporal_state)


        # print(f"Current wp_id: {self.model.wp_id}, N: {N}")
        # print(f"Spatial state: {self.model.spatial_state}")

        # Check if spatial state is None
        if self.model.spatial_state is None:
            print("Spatial state is None. Stopping the vehicle.")
            u = np.array([0.0, 0.0])
            max_delta = 0.0
            return u, max_delta

        # Initialize optimization problem
        self._init_problem(N, self.model.safety_margin)

        # Solve optimization problem
        dec = self.optimizer.solve()

        # Check if optimization was successful
        if dec.x is not None and dec.info.status == 'solved':
            try:
                # Get control signals
                control_signals = np.array(dec.x[-N * nu:])
                # Update delayed inputs
                self.delayed_steering = self.delayed_steering[1:] + [control_signals[1]]
                self.delayed_acceleration = self.delayed_acceleration[1:] + [control_signals[0]]

                control_signals[1::2] = np.arctan(control_signals[1::2] *
                                                  self.model.length)

                # Get current control signal
                u = np.array([control_signals[0], control_signals[1]])

                v = u[0]
                delta = u[1]

                max_delta = np.max(np.abs(control_signals[1::2]))

                # Update control signals
                self.current_control = control_signals

                # Get predicted spatial states
                x = np.reshape(dec.x[:(N + 1) * nx], (N + 1, nx))

                # Update predicted temporal states
                self.current_prediction = self.update_prediction(x, N)

                # if problem solved, reset infeasibility counter
                if self.infeasibility_counter > (N - 1):
                    print(f'Problem solved after {self.infeasibility_counter} infeasible iterations')
                self.infeasibility_counter = 0

            except Exception as e:
                print('Error during processing of control signals:', e)
                # Handle exception if any unexpected error occurs
                u = np.array([0.0, 0.0])
                max_delta = 0.0
                # Update delayed inputs with zeros
                self.delayed_steering = self.delayed_steering[1:] + [u[1]]
                self.delayed_acceleration = self.delayed_acceleration[1:] + [u[0]]
                # Increase infeasibility counter
                self.infeasibility_counter += 1
        else:
            print(f"Optimization failed with status {dec.info.status}")
            print('Infeasible problem. Previously predicted control signal used!')
            id = nu * (self.infeasibility_counter + 1)
            if id + 2 < len(self.current_control):
                u = np.array(self.current_control[id:id + 2])
                max_delta = np.abs(u[1])
            else:
                u = np.array([0.0, 0.0])
                max_delta = 0.0

            # Update delayed inputs
            self.delayed_steering = self.delayed_steering[1:] + [u[1]]
            self.delayed_acceleration = self.delayed_acceleration[1:] + [u[0]]

            # Increase infeasibility counter
            self.infeasibility_counter += 1

        return u, max_delta

    def update_prediction(self, spatial_state_prediction, N):
        """
        Transform the predicted states to predicted x and y coordinates.
        Mainly for visualization purposes.
        :param spatial_state_prediction: list of predicted state variables
        :return: lists of predicted x and y coordinates
        """

        # Containers for x and y coordinates of predicted states
        x_pred, y_pred = [], []

        # Iterate over prediction horizon
        for n in range(2, N):
            # Get associated waypoint
            associated_waypoint = self.model.reference_path.\
                get_waypoint(self.model.wp_id + n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :self.model.n_states])

            # Save predicted coordinates in world coordinate frame
            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred

    def show_prediction(self, ax):
        """
        Display predicted car trajectory on the provided axis.
        :param ax: Matplotlib axis object to plot on
        """

        if self.current_prediction is not None:
            ax.plot(self.current_prediction[0], self.current_prediction[1], c=PREDICTION)
