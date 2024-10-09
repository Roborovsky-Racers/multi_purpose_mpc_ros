import numpy as np
import osqp
from scipy import sparse

class MPCWithDelay:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints,
                 ay_max, use_obstacle_avoidance, use_path_constraints_topic, delay_steps=1):
        """
        Constructor for the Model Predictive Controller with input delay.
        """
        # Parameters
        self.N = N  # horizon
        self.Q = Q  # state cost matrix
        self.R = R  # input cost matrix
        self.QN = QN  # terminal state cost matrix
        self.use_obstacle_avoidance = use_obstacle_avoidance
        self.use_path_constraints_topic = use_path_constraints_topic

        # Model
        self.model = model

        # Dimensions
        self.nx_original = self.model.n_states  # Original state dimension
        self.nu = 2  # Input dimension
        self.delay_steps = delay_steps  # Number of delay steps

        # Augmented state dimension (original states + delayed inputs)
        self.nx = self.nx_original + self.nu * self.delay_steps

        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # Maximum lateral acceleration
        self.ay_max = ay_max

        # Current control and prediction
        self.current_prediction = None
        self.infeasibility_counter = 0

        # Initialize delayed controls
        self.delayed_control = np.zeros(self.nu * self.delay_steps)

        # Initialize Optimization Problem
        self.optimizer = osqp.OSQP()

        # Initialize dynamic constraints if obstacle_avoidance is disabled
        if not self.use_obstacle_avoidance:
            self.model.reference_path.update_simple_path_constraints(
                N,
                self.model.safety_margin)

    def _init_problem(self, N, safety_margin, force_update_dynamic_constraints=False):
        """
        Initialize optimization problem for current time step with delay consideration.
        """
        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']

        # Adjust the state constraints to include delayed inputs
        xmin_aug = np.hstack([xmin, np.tile(umin, self.delay_steps)])
        xmax_aug = np.hstack([xmax, np.tile(umax, self.delay_steps)])

        # Precompute common terms
        nx_N = self.nx * (N + 1)
        nu_N = self.nu * N

        # LTV System Matrices
        A_data = []
        A_row = []
        A_col = []

        B_data = []
        B_row = []
        B_col = []

        # Reference vectors
        xr = np.zeros(nx_N)
        ur = np.zeros(nu_N)
        # Offset for equality constraint
        uq = np.zeros(self.nx * (N + 1))

        # Build system matrices
        for n in range(N):
            # Get linearized system matrices
            current_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            next_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n + 1)
            delta_s = next_waypoint - current_waypoint
            kappa_ref = current_waypoint.kappa
            v_ref = current_waypoint.v_ref

            f, A_lin, B_lin = self.model.linearize(v_ref, kappa_ref, delta_s)

            # Augmented A and B matrices
            A_block = np.zeros((self.nx, self.nx))
            B_block = np.zeros((self.nx, self.nu))

            # Fill A_block
            # Original state dynamics
            A_block[:self.nx_original, :self.nx_original] = A_lin
            # Effect of delayed inputs on original states
            if self.delay_steps > 0:
                A_block[:self.nx_original, self.nx_original:] = np.tile(B_lin, (1, self.delay_steps))

            # Shift delayed inputs
            if self.delay_steps > 1:
                A_block[self.nx_original:-self.nu, self.nx_original+self.nu:] = np.eye(self.nu * (self.delay_steps - 1))

            # Fill B_block
            # Control input affects the last delayed input
            B_block[-self.nu:, :] = np.eye(self.nu)

            # Add A_block and B_block to the big A and B matrices
            idx_x = (n + 1) * self.nx
            idx_u = n * self.nu

            # A matrix
            for i in range(self.nx):
                for j in range(self.nx):
                    A_data.append(A_block[i, j])
                    A_row.append(idx_x + i)
                    A_col.append(n * self.nx + j)

            # B matrix
            for i in range(self.nx):
                for j in range(self.nu):
                    B_data.append(B_block[i, j])
                    B_row.append(idx_x + i)
                    B_col.append(idx_u + j)

            # Reference inputs
            ur[idx_u:idx_u + self.nu] = [v_ref, kappa_ref]

            # Equality constraint offset
            uq[idx_x:idx_x + self.nx_original] = -f

            # Remaining entries are zero (already initialized)

        # Build sparse matrices
        A_eq = sparse.csc_matrix((A_data, (A_row, A_col)), shape=(nx_N, nx_N))
        B_eq = sparse.csc_matrix((B_data, (B_row, B_col)), shape=(nx_N, nu_N))

        # Equality constraints
        Ax = sparse.eye(nx_N) - A_eq
        Aeq = sparse.hstack([Ax, B_eq])

        # Initial state
        x0 = np.hstack([self.model.spatial_state[:], self.delayed_control])

        # Equality constraint bounds
        leq = np.hstack([-x0, uq[self.nx:]])
        ueq = leq

        # Inequality constraints
        xmin_dyn = np.kron(np.ones(N + 1), xmin_aug)
        xmax_dyn = np.kron(np.ones(N + 1), xmax_aug)
        umin_dyn = np.kron(np.ones(N), umin)
        umax_dyn = np.kron(np.ones(N), umax)

        # Path constraints (if applicable)
        # ...（省略：パス制約の更新コード）

        # Total inequality constraints
        lineq = np.hstack([xmin_dyn, umin_dyn])
        uineq = np.hstack([xmax_dyn, umax_dyn])

        # Stack constraints
        Aineq = sparse.eye(nx_N + nu_N)
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Cost function
        Q_block = sparse.block_diag([sparse.kron(sparse.eye(N), self.Q), self.QN,
                                     sparse.csc_matrix((self.nu * self.delay_steps * (N + 1), self.nu * self.delay_steps * (N + 1)))])
        R_block = sparse.kron(sparse.eye(N), self.R)
        P = sparse.block_diag([Q_block, R_block], format='csc')

        q = np.hstack([
            -Q_block.dot(xr),
            -R_block.dot(ur)
        ])

        # Setup optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self):
        """
        Get control signal given the current position of the car.
        """
        # Update current waypoint
        self.model.get_current_waypoint()

        # Update spatial state
        self.model.spatial_state = self.model.t2s(reference_state=self.model.temporal_state,
                                                  reference_waypoint=self.model.current_waypoint)

        # Initialize optimization problem
        self._init_problem(self.N, self.model.safety_margin)

        # Solve optimization problem
        res = self.optimizer.solve()

        if res.info.status != 'solved':
            print('OSQP did not solve the problem!')
            u = self.delayed_control[-self.nu:]
            max_delta = np.abs(u[1])
        else:
            # Extract control inputs
            u = res.x[-self.N * self.nu:][:self.nu]
            print(u)
            max_delta = np.max(np.abs(res.x[-self.N * self.nu + 1::self.nu]))

            # Update delayed controls
            self.delayed_control = np.roll(self.delayed_control, -self.nu)
            self.delayed_control[-self.nu:] = u

            # Update prediction for visualization
            x_pred = res.x[:(self.N + 1) * self.nx]
            x_pred = x_pred.reshape((self.N + 1, self.nx))
            self.current_prediction = self.update_prediction(x_pred[:, :self.nx_original], self.N)

        return u, max_delta

    def update_prediction(self, spatial_state_prediction, N):
        """
        Transform the predicted states to predicted x and y coordinates.
        """
        x_pred, y_pred = [], []

        for n in range(2, N):
            associated_waypoint = self.model.reference_path.get_waypoint(self.model.wp_id + n)
            predicted_temporal_state = self.model.s2t(associated_waypoint,
                                                      spatial_state_prediction[n, :])

            x_pred.append(predicted_temporal_state.x)
            y_pred.append(predicted_temporal_state.y)

        return x_pred, y_pred
