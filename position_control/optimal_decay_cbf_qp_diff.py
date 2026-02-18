import numpy as np
import cvxpy as cp
from qpth.qp import QPFunction, QPSolvers
from networks.policies import QPNetwork

class NotCompatibleError(Exception):
    '''
    Exception raised for errors when the robot model is not compatible with the controller.
    '''

    def __init__(self, message="Currently not compatible with the robot model."):
        self.message = message
        super().__init__(self.message)
        
class OptimalDecayCBFQP:
    def __init__(self, robot, robot_spec, num_obs=1, max_obs_constraints=5):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.max_obs_constraints = max_obs_constraints

        if self.robot_spec['model'] == 'DynamicUnicycle2D': # TODO: not compatible with other robot models yet
            self.cbf_param = {}
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param = {}
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.cbf_param = {}            
            self.cbf_param['alpha'] = 2.0
            self.cbf_param['omega1'] = 1.5  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'Quad2D':
            self.cbf_param = {}            
            self.cbf_param['alpha1'] = 0.5
            self.cbf_param['alpha2'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay
            self.cbf_param['omega2'] = 1.0  # Initial omega
            self.cbf_param['p_sb2'] = 10**4  # Penalty parameter for soft decay
        elif self.robot_spec['model'] == 'Quad3D':
            self.cbf_param = {}            
            self.cbf_param['alpha'] = 0.5
            self.cbf_param['omega1'] = 1.0  # Initial omega
            self.cbf_param['p_sb1'] = 10**4  # Penalty parameter for soft decay         
        else:
            raise NotCompatibleError("Infeasible or Collision")

        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.Q_net = QPNetwork(self.robot_spec, self.max_obs_constraints)
        self.Q = cp
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.omega1 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.omega2 = cp.Variable((1, 1))  # Optimal-decay parameter
        self.A1 = cp.Parameter((self.max_obs_constraints+4, 2), value=np.zeros((self.max_obs_constraints+4, 2)))
        self.b1 = cp.Parameter((self.max_obs_constraints+4, 1), value=np.zeros((self.max_obs_constraints+4, 1)))
        self.h = cp.Parameter((self.max_obs_constraints+4, 1), value=np.zeros((self.max_obs_constraints+4, 1)))
        self.h_dot = cp.Parameter((self.max_obs_constraints+4, 1), value=np.zeros((self.max_obs_constraints+4, 1)))
        
        if self.robot_spec['model'] in ['KinematicBicycle2D_C3BF', 'Quad3D']:
            objective = cp.Minimize(
                cp.sum_squares(self.u - self.u_ref) +
                self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
            )
        else:
            objective = cp.Minimize(
                cp.sum_squares(self.u - self.u_ref) +
                self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1']) +
                self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2'])
            )
        
        # objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref) 
        #                         + self.cbf_param['p_sb1'] * cp.square(self.omega1 - self.cbf_param['omega1'])
        #                         + self.cbf_param['p_sb2'] * cp.square(self.omega2 - self.cbf_param['omega2']))

        if self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['w_max'],
            ]
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2'])* self.omega1 @ self.h_dot +
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['beta_max'],
            ]
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            constraints = [
                self.A1 @ self.u + self.b1 + self.cbf_param['alpha'] * self.h @ self.omega1 >= 0,
                cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                cp.abs(self.u[1]) <= self.robot_spec['beta_max'],
            ]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [
                self.A1 @ self.u + self.b1 + 
                (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * self.omega1 @ self.h_dot + 
                self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * self.h @ self.omega2 >= 0,
                self.u[0] >= self.robot_spec['f_min'],
                self.u[0] <= self.robot_spec['f_max'],
                self.u[1] >= self.robot_spec['f_min'],
                self.u[1] <= self.robot_spec['f_max'],
            ]
        elif self.robot_spec['model'] == 'Quad3D':
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((self.num_obs, 4), value=np.zeros((self.num_obs, 4)))
            self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            self.h = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            self.omega1 = cp.Variable((1, 1))
            constraints = [
                self.A1 @ self.u + self.b1 + self.cbf_param['alpha'] * self.h @ self.omega1 >= 0,
                self.u[0] >= 0.0,
                self.u[0] <= self.robot_spec['f_max'],
                cp.abs(self.u[1]) <= self.robot_spec['phi_dot_max'],
                cp.abs(self.u[2]) <= self.robot_spec['theta_dot_max'],
                cp.abs(self.u[3]) <= self.robot_spec['psi_dot_max'],
            ]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        # Reset constraint matrices to avoid stale values from previous solve
        self.A1.value[:] = 0
        self.b1.value[:] = 0
        self.h.value[:] = 0
        self.h_dot.value[:] = 0
        
        # Update the CBF constraints
        if obs_list is None:
            # All matrices already set to zero above
            pass
        else:
            row_idx = 0
            for i, obs in enumerate(obs_list):
                if obs is None:
                    continue
                
                # Stop if we exceed allocated constraints
                if row_idx >= self.num_obs:
                    break
                
                if self.robot_spec['model'] in ['KinematicBicycle2D_C3BF', 'Quad3D']:
                    h, dh_dx = self.robot.agent_barrier(obs)
                    self.A1.value[row_idx,:] = dh_dx @ self.robot.g()
                    self.b1.value[row_idx,:] = dh_dx @ self.robot.f()
                    self.h.value[row_idx,:] = h
                elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'KinematicBicycle2D', 'Quad2D']:
                    h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                    self.A1.value[row_idx,:] = dh_dot_dx @ self.robot.g()
                    self.b1.value[row_idx,:] = dh_dot_dx @ self.robot.f()
                    self.h.value[row_idx,:] = h
                    self.h_dot.value[row_idx,:] = h_dot
                
                row_idx += 1
            
        h_list, dh_dx_list = self.robot.robot.agent_barrier_walls(robot_state, self.robot_spec['radius'])

        for h, dh_dx in zip(h_list, dh_dx_list):
            if row_idx >= self.num_obs + 4:
                break
            self.A1.value[row_idx,:] = dh_dx @ self.robot.g()
            self.b1.value[row_idx,:] = dh_dx @ self.robot.f()
            self.h.value[row_idx,:] = h
            row_idx += 1

        # print(self.omega1.value, self.omega2.value)

        self.u_ref.value = control_ref.reshape(-1, 1)

        # Solve the optimization problem
        self.cbf_controller.solve(solver=cp.OSQP)
        self.status = self.cbf_controller.status
        print("CBF-QP Solver Status:", self.status)

        if (self.status != "optimal"):
            return self.u_ref.value.flatten()
        else:
            return self.u.value.flatten()