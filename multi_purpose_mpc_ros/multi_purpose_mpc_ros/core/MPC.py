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


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, use_obstacle_avoidance, use_path_constraints_topic):
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

        # Dimensions
        self.nx = self.model.n_states
        self.nu = 2

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
        self.current_control = np.zeros((self.nu*self.N))

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

        # Initialize dynamic constraints once if obstacle_avoidance is disabled
        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N, 
                self.model.length,
                self.model.width,
                self.model.safety_margin)

    def _init_problem(self, N):
        """
        Initialize optimization problem for current time step.
        """

        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # Precompute common terms
        nx_N = self.nx * (N + 1)
        nu_N = self.nu * N

        # LTV System Matrices
        A = np.zeros((nx_N, nx_N))
        B = np.zeros((nx_N, nu_N))

        # Reference vector for state and input variables
        ur = np.zeros(nu_N)
        xr = np.zeros(nx_N)
        # Offset for equality constraint (due to B * (u - ur))
        uq = np.zeros(N * self.nx)

        # Dynamic constraints
        xmin_dyn = np.kron(np.ones(N + 1), xmin)
        xmax_dyn = np.kron(np.ones(N + 1), xmax)
        umax_dyn = np.kron(np.ones(N), umax)
        # Get curvature predictions from previous control signals
        kappa_pred = np.tan(np.array(self.current_control[3:] +
                                     self.current_control[-1:])) / self.model.length

        # Iterate over horizon
        for n in range(N):

            # Get information about current waypoint
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            v_ref = current_waypoint.v_ref

            # Compute LTV matrices
            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)
            A[(n+1) * self.nx: (n+2)*self.nx, n * self.nx:(n+1)*self.nx] = A_lin
            B[(n+1) * self.nx: (n+2)*self.nx, n * self.nu:(n+1)*self.nu] = B_lin

            # Set reference for input signal
            ur[n*self.nu:(n+1)*self.nu] = [v_ref, kappa_ref]
            # Compute equality constraint offset (B*ur)
            uq[n * self.nx:(n+1)*self.nx] = B_lin.dot([v_ref, kappa_ref]) - f

            # Constrain maximum speed based on predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n]) + 1e-12))
            umax_dyn[self.nu*n] = min(vmax_dyn, umax_dyn[self.nu*n])

        # Update path constraints
        if self.use_obstacle_avoidance and not self.use_path_constraints_topic:
            ub, lb, _ = self.model.reference_path.update_path_constraints(
                self.model.wp_id + 1,
                [self.model.temporal_state.x, self.model.temporal_state.y, self.model.temporal_state.psi],
                N,
                self.model.length,
                self.model.width,
                self.model.safety_margin)
        else:
            ub = self.model.reference_path.path_constraints[0][self.model.wp_id]
            lb = self.model.reference_path.path_constraints[1][self.model.wp_id]
            self.model.reference_path.border_cells.current_wp_id = self.model.wp_id

        # Update dynamic state constraints
        xmin_dyn[0] = xmax_dyn[0] = self.model.spatial_state.e_y
        xmin_dyn[self.nx::self.nx] = lb
        xmax_dyn[self.nx::self.nx] = ub

        # Set reference for state as center-line of drivable area
        xr[self.nx::self.nx] = (lb + ub) / 2

        # Get equality matrix
        Ax = sparse.kron(sparse.eye(N + 1),
                         -sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Get inequality matrix
        Aineq = sparse.eye(nx_N + nu_N)
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Construct bounds
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq
        lineq = np.hstack([xmin_dyn, np.kron(np.ones(N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        P = sparse.block_diag([sparse.kron(sparse.eye(N), self.Q), self.QN,
                               sparse.kron(sparse.eye(N), self.R)], format='csc')
        q = np.hstack(
            [-np.tile(np.diag(self.Q.toarray()), N) * xr[:-self.nx],
             -self.QN.dot(xr[-self.nx:]),
             -np.tile(np.diag(self.R.toarray()), N) * ur])

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self) -> Tuple[np.ndarray, float]:
        """
        Get control signal given the current position of the car. Solves a
        finite time optimization problem based on the linearized car model.
        """

        # Number of state variables
        nx = self.model.n_states
        nu = 2

        # Update current waypoint
        self.model.get_current_waypoint()

        # check if the horizon exceeds the number of waypoints
        N = self.N
        if self.model.wp_id >= self.model.reference_path.n_waypoints and not self.model.reference_path.circular:
            N = self.model.reference_path.n_waypoints - self.model.wp_id
            # print('Horizon exceeds number of waypoints. ')
        # print(f"N: {N}")

        # Update spatial state
        self.model.spatial_state = self.model.t2s(reference_state=
            self.model.temporal_state, reference_waypoint=
            self.model.current_waypoint)

        # Initialize optimization problem
        self._init_problem(N)

        # Solve optimization problem
        dec = self.optimizer.solve()

        try:
            # Get control signals
            control_signals = np.array(dec.x[-N*nu:])
            control_signals[1::2] = np.arctan(control_signals[1::2] *
                                              self.model.length)
            v = control_signals[0]
            delta = control_signals[1]

            # max delta in prediction horizon
            max_delta = np.max(np.abs(control_signals[1:len(control_signals) // 3 * 2:2]))

            # Update control signals
            self.current_control = control_signals

            # Get predicted spatial states
            x = np.reshape(dec.x[:(N+1)*nx], (N+1, nx))

            # Update predicted temporal states
            self.current_prediction = self.update_prediction(x, N)

            # Get current control signal
            u = np.array([v, delta])

            # if problem solved, reset infeasibility counter
            self.infeasibility_counter = 0

        except Exception as e:
            # print('Infeasible problem. Previously predicted'
            #       ' control signal used!')
            id = nu * (self.infeasibility_counter + 1)
            u = np.array(self.current_control[id:id+2])
            max_delta = np.abs(u[1])

            # increase infeasibility counter
            self.infeasibility_counter += 1

        # if self.infeasibility_counter == (N - 1):
        #     print('No control signal computed!')
        #     exit(1)

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
                get_waypoint(self.model.wp_id+n)
            # Transform predicted spatial state to temporal state
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                            spatial_state_prediction[n, :])

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
            # ax.scatter(self.current_prediction[0], self.current_prediction[1],
            #            c=PREDICTION, s=5)
            ax.plot(self.current_prediction[0], self.current_prediction[1], c=PREDICTION)

