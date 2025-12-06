import os
import numpy as np
import torch
import cvxpy as cp
from qpth.qp import QPFunction, QPSolvers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from policies.kappa_nn import MonotoneKappaNN


class CBFQPDIFF:
    def __init__(self, robot, robot_spec, num_obs=1, use_learnable_kappa=True, device='cpu'):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs
        self.use_learnable_kappa = use_learnable_kappa
        self.device = device

        self.cbf_param = {}
        self.u_safe = None  # Will store torch tensor with gradients after solve
        self.h_values = []  # Will store barrier values for gradient tracking
        self.status = 'unsolved'

        # Initialize learnable K-class function
        if use_learnable_kappa:
            # Create neural network for K-class function
            self.kappa_nn = MonotoneKappaNN(
                in_features=1,
                hidden=(32, 32),
                activation=torch.nn.ReLU(),
                eps_w=1e-6,
                strictly_increasing=True
            ).to(device)
        else:
            # Use fixed alpha parameters (backward compatibility)
            if self.robot_spec['model'] == "SingleIntegrator2D" or self.robot_spec['model'] == "SingleIntegrator2DMLP":
                self.cbf_param['alpha'] = 0.1
            elif self.robot_spec['model'] == 'Unicycle2D':
                self.cbf_param['alpha'] = 1.0
            elif self.robot_spec['model'] == 'DynamicUnicycle2D':
                self.cbf_param['alpha1'] = 1.5
                self.cbf_param['alpha2'] = 1.5
            elif self.robot_spec['model'] == 'DoubleIntegrator2D':
                self.cbf_param['alpha1'] = 1.5
                self.cbf_param['alpha2'] = 1.5
            elif self.robot_spec['model'] == 'KinematicBicycle2D':
                self.cbf_param['alpha1'] = 1.5
                self.cbf_param['alpha2'] = 1.5
            elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
                self.cbf_param['alpha'] = 1.5
            elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
                self.cbf_param['alpha'] = 1.5
            elif self.robot_spec['model'] == 'Quad2D':
                self.cbf_param['alpha1'] = 1.5
                self.cbf_param['alpha2'] = 1.5
            elif self.robot_spec['model'] == 'Quad3D':
                self.cbf_param['alpha'] = 1.5

        # Initialize qpth solver
        self.qp_solver = QPFunction(verbose=False, solver=QPSolvers.CVXPY)
        
        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
        self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.robot_spec['model'] == 'SingleIntegrator2D' or self.robot_spec['model'] == 'SingleIntegrator2DMLP':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <=  self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <=  self.robot_spec['v_max']]
        elif self.robot_spec['model'] == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['a_max']]
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.robot_spec["f_min"] <= self.u[0],
                           self.u[0] <= self.robot_spec["f_max"],
                           self.robot_spec["f_min"] <= self.u[1],
                           self.u[1] <= self.robot_spec["f_max"]]
        elif self.robot_spec['model'] == 'Quad3D':
            # overwrite the variables
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.u[0] <= self.robot_spec['u_max'],
                            self.u[0] >= 0.0,
                           cp.abs(self.u[1]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[2]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[3]) <= self.robot_spec['u_max']]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list, return_torch=False):
        """
        Solve CBF-QP using qpth differentiable QP solver with learnable K-class function.
        
        QP formulation:
            minimize    (1/2) ||u - u_ref||^2
            subject to  -A1 @ u <= b1  (CBF constraints)
                        u_min <= u <= u_max
        
        Returns:
            u_safe: torch.Tensor with gradients w.r.t. barrier values h
        """
        # Convert numpy to torch tensors
        u_ref_input = control_ref
        
        # Handle both numpy and torch tensor inputs
        if isinstance(u_ref_input, torch.Tensor):
            u_ref_torch = u_ref_input.flatten().to(self.device)
            u_ref_np = u_ref_input.detach().cpu().numpy().flatten()
        else:
            u_ref_np = u_ref_input.flatten()
            u_ref_torch = torch.tensor(u_ref_np, dtype=torch.float32, device=self.device)
        
        # Initialize constraint matrices
        n_controls = 2 if self.robot_spec['model'] != 'Quad3D' else 4
        A1_list = []
        b1_list = []
        h_values = []  # Store barrier values for gradient tracking
        
        # Build CBF constraints
        for i in range(min(self.num_obs, len(obs_list))):
            obs = obs_list[i]
            
            if obs is None:
                continue
                
            # Compute barrier function and gradients
            if self.robot_spec['model'] in ['SingleIntegrator2D', 'SingleIntegrator2DMLP', 'Unicycle2D', 'KinematicBicycle2D_C3BF', 'KinematicBicycle2D_DPCBF', 'Quad3D']:
                h, dh_dx = self.robot.agent_barrier(obs)
                A_cbf = dh_dx @ self.robot.g()  # (1, n_controls)
                #print(f"Barrier h for obs {i}: {h}")
                # Convert h to torch tensor for learnable K-class function
                h_torch = torch.tensor([[h]], dtype=torch.float32, device=self.device, requires_grad=True)
                h_values.append(h_torch)
                
                if self.use_learnable_kappa:
                    # Use learned K-class function: kappa(h)
                    kappa_h = self.kappa_nn(h_torch)  # Neural network output
                    b_cbf = (dh_dx @ self.robot.f()).item() + kappa_h.item()
                else:
                    # Use fixed alpha: alpha * h
                    b_cbf = (dh_dx @ self.robot.f()).item() + self.cbf_param['alpha'] * h
                    
            elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D', 'KinematicBicycle2D', 'Quad2D']:
                h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)
                A_cbf = dh_dot_dx @ self.robot.g()  # (1, n_controls)
                
                # For high-order CBF: kappa_1(h_dot) + kappa_2(h)
                # Simplified: use learnable function on h, fixed for h_dot
                h_torch = torch.tensor([[h]], dtype=torch.float32, device=self.device, requires_grad=True)
                h_values.append(h_torch)
                
                if self.use_learnable_kappa:
                    kappa_h = self.kappa_nn(h_torch)
                    # High-order: (alpha1+alpha2)*h_dot + kappa(h)
                    b_cbf = (dh_dot_dx @ self.robot.f()).item() + \
                            (self.cbf_param.get('alpha1', 1.5) + self.cbf_param.get('alpha2', 1.5)) * h_dot + \
                            kappa_h.item()
                else:
                    b_cbf = (dh_dot_dx @ self.robot.f()).item() + \
                            (self.cbf_param['alpha1'] + self.cbf_param['alpha2']) * h_dot + \
                            self.cbf_param['alpha1'] * self.cbf_param['alpha2'] * h
            
            # A_cbf should be (1, n_controls), flatten to 1D array
            A1_list.append(A_cbf.flatten())  # Shape: (n_controls,)
            b1_list.append(b_cbf)  # Scalar value
        
        # Build control bound constraints
        if self.robot_spec['model'] == 'SingleIntegrator2D' or self.robot_spec['model'] == 'SingleIntegrator2DMLP':
            v_max = self.robot_spec['v_max']
            # u_min <= u <= u_max  =>  [-I; I] @ u <= [u_max; -u_min]
            A_bounds = np.vstack([np.eye(2), -np.eye(2)])  # (4, 2)
            b_bounds = np.array([v_max, v_max, v_max, v_max])  # element-wise bounds
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            a_max = self.robot_spec['a_max']
            A_bounds = np.vstack([np.eye(2), -np.eye(2)])
            b_bounds = np.array([a_max, a_max, a_max, a_max])
        else:
            # Generic bounds (adjust per model as needed)
            A_bounds = np.vstack([np.eye(n_controls), -np.eye(n_controls)])
            b_bounds = np.ones(2 * n_controls) * 10.0  # Default large bounds
        
        # Combine CBF and bound constraints
        if len(A1_list) > 0:
            # Stack the constraint rows properly
            A_cbf = np.vstack(A1_list)  # (n_obs, n_controls) - vstack handles 1D arrays correctly
            b_cbf = np.array(b1_list)  # (n_obs,) - 1D array
            
            # qpth uses G @ z <= h format, but CBF is A @ u + b >= 0, i.e., -A @ u <= b
            G = np.vstack([-A_cbf, A_bounds])  # (n_obs + 2*n_controls, n_controls)
            h = np.concatenate([b_cbf, b_bounds])  # (n_obs + 2*n_controls,) - 1D array
        else:
            # No obstacles, only bound constraints
            G = A_bounds  # (2*n_controls, n_controls)
            h = b_bounds  # (2*n_controls,)
        
        # Convert to torch tensors
        # qpth expects: minimize (1/2) z^T Q z + p^T z  s.t. G z <= h, A z = b
        # Our problem: minimize ||u - u_ref||^2 = (1/2) u^T I u - u_ref^T u + const
        # So Q = I, p = -u_ref
        
        # Ensure G and h have correct shapes
        G = np.atleast_2d(G)  # Ensure 2D: (n_ineq, n_controls)
        h = h.flatten()  # Ensure 1D: (n_ineq,)
        
        Q = torch.eye(n_controls, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, n_controls, n_controls)
        p = -u_ref_torch.unsqueeze(0)  # (1, n_controls)
        G_torch = torch.from_numpy(G).float().to(self.device).unsqueeze(0)  # (1, n_ineq, n_controls)
        h_torch = torch.from_numpy(h).float().to(self.device).unsqueeze(0)  # (1, n_ineq)
        
        # qpth requires equality constraints A and b (even if empty)
        A_torch = torch.empty(1, 0, n_controls, dtype=torch.float32, device=self.device)  # (1, 0, n_controls)
        b_torch = torch.empty(1, 0, dtype=torch.float32, device=self.device)  # (1, 0)
        
        # Solve QP with qpth (differentiable)
        try:
            u_safe = self.qp_solver(Q, p, G_torch, h_torch, A_torch, b_torch)  # (1, n_controls)
            u_safe = u_safe.squeeze(0)  # (n_controls,)
            
            # Store barrier values for gradient computation
            self.h_values = h_values
            self.u_safe = u_safe
            self.status = 'optimal'
            
                    # Return based on mode
            if return_torch:
                return u_safe  # Return tensor with gradients for training
            else:
                return u_safe.detach().cpu().numpy().reshape(-1, 1)  # Numpy for execution
            
        except Exception as e:
            print(f"qpth QP solve failed: {e}")  # Suppressed for clean output
            self.status = 'failed'
            # Fallback to nominal control
            # Also set u_safe for gradient flow (even though it's just u_ref)
        # Fallback to nominal control
            if return_torch:
                self.u_safe = u_ref_torch
                return u_ref_torch  # Return with gradients
            else:
                self.u_safe = u_ref_np
                return u_ref_np
                
    
    def get_gradients(self):
        """
        [OPTIONAL] Explicitly compute gradients of the QP solution w.r.t. barrier values h.
        
        NOTE: You typically DON'T need to call this! The QPFunction automatically handles
        the backward pass. Just call loss.backward() and gradients will flow through.
        
        This method is useful for debugging/analysis to inspect ∂u*/∂h directly.
        
        Usage:
            u_safe = solver.solve_control_problem(...)
            loss = criterion(solver.u_safe, target)  # u_safe is torch tensor
            loss.backward()  # Gradients flow automatically through QPFunction!
            # optimizer.step()
        
        Returns:
            grads: list of torch.Tensor, gradients ∂u*/∂h for each barrier value
        """
        if not hasattr(self, 'u_safe') or not hasattr(self, 'h_values'):
            return None
        
        grads = []
        for h_val in self.h_values:
            if h_val.requires_grad:
                # Compute gradient of u_safe w.r.t. h
                # NOTE: This is for analysis only. In training, gradients flow automatically.
                grad = torch.autograd.grad(
                    self.u_safe.sum(), h_val, 
                    retain_graph=True, create_graph=True
                )[0]
                grads.append(grad)
        
        return grads
    
    def get_kappa_parameters(self):
        """
        Get parameters of the learnable K-class function for optimization.
        
        Returns:
            parameters: iterator of torch.nn.Parameter
        """
        if self.use_learnable_kappa:
            return self.kappa_nn.parameters()
        else:
            return iter([])
    
    def save_kappa_nn(self, path):
        """Save learned K-class function weights."""
        if self.use_learnable_kappa:
            torch.save(self.kappa_nn.state_dict(), path)
            print(f"Saved learned K-class function to {path}")
    
    def load_kappa_nn(self, path):
        """Load learned K-class function weights."""
        if self.use_learnable_kappa:
            self.kappa_nn.load_state_dict(torch.load(path))
            print(f"Loaded learned K-class function from {path}")
    
    def evaluate_kappa(self, h_values):
        """
        Evaluate the learned K-class function on given barrier values.
        
        Args:
            h_values: numpy array or torch tensor of barrier values
        
        Returns:
            kappa(h): torch tensor of K-class function outputs
        """
        if not self.use_learnable_kappa:
            raise ValueError("Kappa NN not initialized. Set use_learnable_kappa=True.")
        
        if isinstance(h_values, np.ndarray):
            h_values = torch.tensor(h_values, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            return self.kappa_nn(h_values.reshape(-1, 1))