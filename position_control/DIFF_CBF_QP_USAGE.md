# Differentiable CBF-QP with Learnable K-Class Function

## Overview

The updated `cbf_qp_diff.py` now uses:
1. **qpth** - A differentiable QP solver that allows gradients to flow through the optimization
2. **MonotoneKappaNN** - A learnable neural network K-class function that replaces the fixed `alpha * h`

## Key Changes

### 1. Initialization

```python
from position_control.cbf_qp_diff import CBFQPDIFF

# Create differentiable CBF-QP with learnable K-class function
pos_controller = CBFQP(
    robot, 
    robot_spec, 
    num_obs=5,
    use_learnable_kappa=True,  # Enable learnable K-class function
    device='cuda'              # or 'cpu'
)
```

### 2. K-Class Function

**Before (fixed alpha):**
```python
# Linear K-class: kappa(h) = alpha * h
b_cbf = dh/dx @ f(x) + alpha * h
```

**After (learned neural network):**
```python
# Learned monotone K-class: kappa(h) = NN(h)
kappa_h = self.kappa_nn(h_torch)
b_cbf = dh/dx @ f(x) + kappa_h
```

The neural network:
- ✅ **Monotone increasing**: Enforced via positive weights
- ✅ **kappa(0) = 0**: Enforced by construction
- ✅ **Strictly increasing**: Optional residual term `alpha * h`

### 3. QP Formulation

**qpth expects:**
```
minimize    (1/2) z^T Q z + p^T z
subject to  G z <= h
```

**Our CBF-QP:**
```
minimize    ||u - u_ref||^2
subject to  dh/dx @ (f + g@u) + kappa(h) >= 0  (CBF)
            u_min <= u <= u_max
```

**Conversion:**
- `Q = I` (identity matrix)
- `p = -u_ref`
- `G = [-A_cbf; I; -I]` (inequality matrix)
- `h = [b_cbf; u_max; -u_min]` (inequality bounds)

### 4. Automatic Gradient Flow

**qpth's QPFunction automatically handles the backward pass!** You don't need to manually compute gradients.

```python
# Solve QP (returns numpy, but stores torch tensor internally)
u_safe_np = pos_controller.solve_control_problem(robot_state, control_ref, obs_list)

# Get the torch tensor for gradient tracking
u_safe_torch = pos_controller.u_safe  # This is a torch.Tensor with gradients

# Define your loss (e.g., track expert control)
loss = torch.nn.functional.mse_loss(u_safe_torch, u_expert)

# Gradients flow automatically through QPFunction backward!
loss.backward()  
# Now kappa_nn parameters have gradients: ∂loss/∂θ_kappa
# Ready for optimizer.step()
```

**Optional:** For debugging/analysis, you can explicitly compute `∂u*/∂h`:
```python
grads = pos_controller.get_gradients()  # Returns list of ∂u*/∂h_i
# But you typically don't need this - just use loss.backward()
```

## Training the Learnable K-Class Function

### Example Training Loop

```python
import torch
import torch.optim as optim

# Get learnable parameters
kappa_params = pos_controller.get_kappa_parameters()
optimizer = optim.Adam(kappa_params, lr=1e-3)

# Training loop
for episode in range(num_episodes):
    # Collect trajectory
    states, controls, rewards = [], [], []
    
    for step in range(max_steps):
        # Solve differentiable CBF-QP
        u_safe = pos_controller.solve_control_problem(
            robot_state, control_ref, obs_list
        )
        
        # Convert to torch for gradient tracking
        u_safe_torch = pos_controller.u_safe  # Already a torch tensor
        
        # Execute control
        next_state = robot.step(u_safe.numpy())
        reward = compute_reward(next_state, goal)
        
        states.append(robot_state)
        controls.append(u_safe_torch)
        rewards.append(reward)
    
    # Compute loss (example: track expert trajectory)
    loss = 0
    for u_safe, u_expert in zip(controls, expert_controls):
        loss += torch.nn.functional.mse_loss(u_safe, u_expert)
    
    # Backprop through the QP and kappa function
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Episode {episode}, Loss: {loss.item()}")

# Save learned K-class function
pos_controller.save_kappa_nn('kappa_weights.pth')
```

### What Gets Learned

The neural network learns an **adaptive safety margin**:
- More conservative (larger `kappa(h)`) when close to obstacles (small `h`)
- Less conservative when far from obstacles (large `h`)
- Optimized end-to-end for the task (tracking, goal-reaching, etc.)

## Backward Compatibility

To use the original fixed-alpha version:

```python
pos_controller = CBFQP(
    robot, 
    robot_spec,
    use_learnable_kappa=False  # Use fixed alpha parameters
)
```

This will use the standard linear K-class functions from the original code.

## Usage with RL (SAC/TD3)

### Integration Example

```python
class CBFRLAgent:
    def __init__(self, robot, robot_spec):
        # Create differentiable CBF-QP
        self.cbf_qp = CBFQP(
            robot, robot_spec,
            use_learnable_kappa=True,
            device='cuda'
        )
        
        # Add kappa parameters to optimizer
        self.kappa_optimizer = optim.Adam(
            self.cbf_qp.get_kappa_parameters(),
            lr=3e-4
        )
    
    def select_action(self, state, obs_list):
        # Policy network outputs nominal control
        u_nom = self.policy_network(state)
        
        # CBF-QP filters it for safety
        u_safe = self.cbf_qp.solve_control_problem(
            state, 
            {'u_ref': u_nom.detach().numpy()},
            obs_list
        )
        
        return u_safe
    
    def update(self, batch):
        # SAC/TD3 update for policy
        policy_loss = self.compute_policy_loss(batch)
        
        # Additional loss to learn better K-class function
        # Example: minimize intervention (u_safe close to u_nom)
        kappa_loss = 0
        for transition in batch:
            u_nom = self.policy_network(transition.state)
            u_safe_torch = self.cbf_qp.u_safe  # From last solve
            kappa_loss += torch.nn.functional.mse_loss(u_safe_torch, u_nom)
        
        # Update kappa function
        self.kappa_optimizer.zero_grad()
        kappa_loss.backward()
        self.kappa_optimizer.step()
```

## Advantages

1. ✅ **End-to-end learning**: Gradients flow from task loss → QP solution → kappa function → parameters
2. ✅ **Adaptive safety**: Learn task-specific safety margins
3. ✅ **Monotonicity guaranteed**: Neural network structure ensures valid K-class function
4. ✅ **Backward compatible**: Can still use fixed alpha parameters

## Troubleshooting

### qpth installation
```bash
pip install qpth
```

### CUDA issues
If you get CUDA errors, use CPU:
```python
pos_controller = CBFQP(robot, robot_spec, device='cpu')
```

### QP infeasibility
If the QP fails to solve, the code falls back to the nominal control:
```python
self.status = 'failed'
return u_ref_np  # Fallback
```

Check `pos_controller.status` after solving to verify success.

## References

- **qpth**: [Amos & Kolter, 2017 - OptNet: Differentiable Optimization as a Layer in Neural Networks](https://arxiv.org/abs/1703.00443)
- **Learnable CBF**: [Choi et al., 2020 - Reinforcement Learning for Safety-Critical Control under Model Uncertainty](https://arxiv.org/abs/1910.10907)
