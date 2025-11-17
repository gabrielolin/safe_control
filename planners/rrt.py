"""
RRT planner with dynamic feasibility constraints based on robot dynamics
Compatible with LocalTrackingController interface
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches


class RRT:
    """
    RRT planner with dynamic feasibility constraints based on robot model
    """
    
    class Node:
        def __init__(self, state):
            self.state = np.array(state).reshape(-1, 1)  # Column vector
            self.parent = None
            self.control = None  # Control input that led to this node
            self.cost = 0.0
    
    def __init__(self, X0, robot_spec, dt=0.05, 
                 show_animation=False,
                 ax=None, fig=None, env=None,
                 max_iter=2000, goal_sample_rate=0.15, 
                 expand_dis=0.3, path_resolution=0.05, goal_threshold=0.5):
        """
        Initialize RRT planner with same interface as LocalTrackingController
        
        Args:
            X0: Initial state
            robot_spec: Robot specification dictionary
            controller_type: Controller type (not used by planner but kept for compatibility)
            dt: Time step for integration
            show_animation: Whether to show animation
            raise_error: Raise error on failure
            ax: Matplotlib axis for plotting
            fig: Matplotlib figure
            env: Environment handler
            max_iter: Maximum RRT iterations
            goal_sample_rate: Probability of sampling goal
            expand_dis: Distance to extend tree
            path_resolution: Resolution for collision
            goal_threshold: Distance threshold for reaching goal
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.show_animation = show_animation
        self.ax = ax
        self.fig = fig
        self.env = env
        
        # RRT parameters
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_threshold = goal_threshold
        
        # Extract dynamics properties from robot_spec
        self.model = robot_spec['model']
        if self.model == 'SingleIntegrator2D' or self.model == 'SingleIntegrator2DOpenLoop':
            self.state_dim = 2  # [x, y]
            self.control_dim = 2  # [vx, vy]
            self.v_max = robot_spec.get('v_max', 1.0)
        elif self.model == 'DoubleIntegrator2D':
            self.state_dim = 4  # [x, y, vx, vy]
            self.control_dim = 2  # [ax, ay]
            self.v_max = robot_spec.get('v_max', 1.0)
            self.a_max = robot_spec.get('a_max', 1.0)
        elif self.model in ['Unicycle2D', 'DynamicUnicycle2D']:
            self.state_dim = 3  # [x, y, theta]
            self.control_dim = 2  # [v, w] or [a, w]
            self.v_max = robot_spec.get('v_max', 1.0)
            self.w_max = robot_spec.get('w_max', 1.0)
        else:
            # Default to 2D position
            self.state_dim = 2
            self.control_dim = 2
            self.v_max = robot_spec.get('v_max', 1.0)
        
        self.robot_radius = robot_spec.get('radius', 0.25)
        
        # Initialize robot dynamics
        self._setup_robot(X0)
        
        # Tree storage
        self.node_list = []
        self.obstacles = []
        self.path = None
        
    def _setup_robot(self, X0):
        """Setup robot dynamics object"""
        from robots.robot import BaseRobot
        self.robot = BaseRobot(X0.reshape(-1, 1), self.robot_spec, self.dt, self.ax)
    
    def plan(self, start, goal, obstacle_list):
        """
        Plan a dynamically feasible path from start to goal
        
        Args:
            start: Start state (shape depends on robot model)
            goal: Goal position [x, y] or state
            obstacle_list: List of obstacles [[ox, oy, radius, ...], ...]
            animation: Whether to show animation
            
        Returns:
            waypoints: numpy array of shape (N, 2) with waypoints [[x, y], ...]
            controls: list of controls, where controls[i] is applied at waypoints[i] to reach waypoints[i+1]
            Returns (None, None) if no path found
        """
        # Switch to interactive backend for RRT visualization
        original_backend = matplotlib.get_backend()
        if self.show_animation and original_backend.lower() == 'agg':
            # Save current figure number before switching
            old_fig_num = self.fig.number if self.fig is not None else None
            
            # Switch to interactive backend
            matplotlib.use('TkAgg', force=True)
            plt.ion()
            print(f"RRT: Switched from {original_backend} to TkAgg for interactive visualization")
            
            # Recreate the figure with TkAgg backend, copying the old figure's content
            if old_fig_num is not None and self.fig is not None:
                # Get the old figure's renderer and content
                new_fig = plt.figure(figsize=self.fig.get_size_inches())
                new_ax = new_fig.add_subplot(111)
                
                # Copy axes properties
                new_ax.set_xlim(self.ax.get_xlim())
                new_ax.set_ylim(self.ax.get_ylim())
                new_ax.set_xlabel(self.ax.get_xlabel())
                new_ax.set_ylabel(self.ax.get_ylabel())
                new_ax.set_aspect(self.ax.get_aspect())
                
                # Copy all the plot elements (obstacles, etc.)
                for child in self.ax.get_children():
                    if isinstance(child, patches.Circle):
                        new_ax.add_patch(patches.Circle(
                            child.center, child.radius,
                            edgecolor=child.get_edgecolor(),
                            facecolor=child.get_facecolor(),
                            fill=child.get_fill(),
                            alpha=child.get_alpha()
                        ))
                    elif isinstance(child, patches.Rectangle):
                        new_ax.add_patch(patches.Rectangle(
                            child.get_xy(), child.get_width(), child.get_height(),
                            edgecolor=child.get_edgecolor(),
                            facecolor=child.get_facecolor(),
                            fill=child.get_fill(),
                            alpha=child.get_alpha()
                        ))
                
                # Update references
                self.fig = new_fig
                self.ax = new_ax
                plt.show(block=False)
        
        # Initialize
        self.start = self.Node(self._state_to_node(start))
        self.goal = self.Node(self._state_to_node(goal))
        self.node_list = [self.start]
        self.obstacles = obstacle_list if obstacle_list is not None else []
        
        if self.show_animation:
            self._draw_graph()
        
        for i in range(self.max_iter):
            # Sample random state or goal
            if np.random.rand() < self.goal_sample_rate:
                rnd_state = self.goal.state[:self.state_dim, 0]
            else:
                rnd_state = self._sample_random_state()
            
            # Find nearest node in tree
            nearest_node = self._get_nearest_node(rnd_state)
            
            # Steer toward random state using dynamics
            new_node = self._steer(nearest_node, rnd_state)
            
            if new_node is None:
                continue
            
            # Check collision
            if self._check_collision(new_node):
                self.node_list.append(new_node)
                
                # Check if goal is reached
                if self._calc_dist_to_goal(new_node.state[:2, 0]) <= self.goal_threshold:
                    print(f"RRT: Goal reached in {i} iterations!")
                    waypoints, controls = self._generate_final_path(new_node)
                    if self.show_animation:
                        self._draw_final_path(waypoints)
                        print("RRT: Planning complete. Close the window to continue...")
                        # Keep window open longer to show final path
                        plt.pause(2.0)
                        # Switch back to original backend after planning
                        if original_backend.lower() == 'agg':
                            plt.ioff()
                            matplotlib.use('Agg', force=True)
                            print(f"RRT: Switched back to {original_backend}")
                    return waypoints, controls
                
                if self.show_animation and i % 50 == 0:
                    self._draw_graph(new_node)
        
        print(f"RRT: Max iterations reached ({self.max_iter}), no path found")
        if self.show_animation:
            # Switch back to original backend
            if original_backend.lower() == 'agg':
                plt.ioff()
                matplotlib.use('Agg', force=True)
                print(f"RRT: Switched back to {original_backend}")
        return None, None
    
    def _state_to_node(self, state):
        """Convert state to node representation"""
        state = np.array(state).flatten()
        if self.model == 'SingleIntegrator2D' or self.model == 'SingleIntegrator2DOpenLoop':
            # Extract [x, y] from state (may include theta)
            return state[:2]
        elif self.model == 'DoubleIntegrator2D':
            # Extract [x, y, vx, vy] from state (may include theta)
            if len(state) == 2:
                return np.array([state[0], state[1], 0.0, 0.0])
            elif len(state) >= 4:
                return state[:4]
            else:
                return np.hstack([state[:2], [0.0, 0.0]])
        else:
            return state[:self.state_dim]
    
    def _sample_random_state(self):
        """Sample a random state in the workspace"""
        # Assuming workspace bounds (adjust as needed)
        x_min, x_max = 0, 14
        y_min, y_max = 0, 14
        
        if self.model == 'SingleIntegrator2D' or self.model == 'SingleIntegrator2DOpenLoop':
            return np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])
        elif self.model == 'DoubleIntegrator2D':
            return np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(-self.v_max*0.5, self.v_max*0.5),
                np.random.uniform(-self.v_max*0.5, self.v_max*0.5)
            ])
        elif self.model in ['Unicycle2D', 'DynamicUnicycle2D']:
            return np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(-np.pi, np.pi)
            ])
        else:
            return np.random.uniform([x_min, y_min], [x_max, y_max])
    
    def _get_nearest_node(self, rnd_state):
        """Find nearest node in tree to random state"""
        dists = [np.linalg.norm(node.state[:2, 0] - rnd_state[:2]) 
                 for node in self.node_list]
        min_idx = np.argmin(dists)
        return self.node_list[min_idx]
    
    def _steer(self, from_node, to_state, n_steps=10):
        """
        Steer from from_node toward to_state using robot dynamics
        Returns new node if successful, None otherwise
        """
        # Compute control input to steer toward target
        current_state = from_node.state.copy()
        
        # Compute desired control
        if self.model == 'SingleIntegrator2D' or self.model == 'SingleIntegrator2DOpenLoop':
            # Direct velocity control
            direction = to_state[:2] - current_state[:2, 0]
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                return np.array([0.0, 0.0]).reshape(-1, 1)
            direction = direction / dist
            u = direction * self.v_max
            
        elif self.model == 'DoubleIntegrator2D':
            # Acceleration control
            pos_error = to_state[:2] - current_state[:2, 0]
            vel_des = pos_error * 2.0  # Proportional gain
            vel_des_norm = np.linalg.norm(vel_des)
            if vel_des_norm > self.v_max:
                vel_des = vel_des / vel_des_norm * self.v_max
            
            vel_error = vel_des - current_state[2:4, 0]
            u = vel_error * 2.0  # Proportional gain
            u_norm = np.linalg.norm(u)
            if u_norm > self.a_max:
                u = u / u_norm * self.a_max
        else:
            # Default: simple proportional control
            direction = to_state[:2] - current_state[:2, 0]
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                return None
            u = direction / dist * self.expand_dis
        
        # Integrate dynamics for multiple steps
        u = u.reshape(-1, 1)
        new_state = current_state.copy()
        
        for _ in range(n_steps):
            # Use robot dynamics: dx/dt = f(x) + g(x) * u
            # Note: robot dynamics are accessed through self.robot.robot
            f_x = self.robot.robot.f(new_state)
            g_x = self.robot.robot.g(new_state)
            new_state = new_state + (f_x + g_x @ u) * self.dt
        
        # Create new node
        new_node = self.Node(new_state)
        new_node.parent = from_node
        new_node.control = u
        new_node.cost = from_node.cost + np.linalg.norm(new_state[:2] - current_state[:2])
        
        return new_node
    
    def _check_collision(self, node):
        """Check if node is collision-free"""
        if node is None:
            return False
        
        # Check collision with obstacles
        pos = node.state[:2, 0]
        
        for obs in self.obstacles:
            if len(obs) < 3:
                continue
            
            ox, oy, r_obs = obs[0], obs[1], obs[2]
            
            # Check if obstacle is circle (flag == 0 or not specified)
            if len(obs) <= 6 or obs[6] == 0:
                # Circle obstacle
                dist = np.linalg.norm(pos - np.array([ox, oy]))
                if dist < (r_obs + self.robot_radius):
                    return False
            else:
                # Superellipse obstacle
                if len(obs) >= 6:
                    a, b, e, theta = obs[2], obs[3], obs[4], obs[5]
                    # Transform to obstacle frame
                    dx = pos[0] - ox
                    dy = pos[1] - oy
                    pox_prime = np.cos(theta) * dx + np.sin(theta) * dy
                    poy_prime = -np.sin(theta) * dx + np.cos(theta) * dy
                    
                    h = (pox_prime / (a + self.robot_radius))**e + \
                        (poy_prime / (b + self.robot_radius))**e - 1
                    if h <= 0:
                        return False
        
        # Check workspace bounds
        if pos[0] < 0 or pos[0] > 14 or pos[1] < 0 or pos[1] > 14:
            return False
        
        return True
    
    def _calc_dist_to_goal(self, pos):
        """Calculate distance to goal"""
        return np.linalg.norm(pos - self.goal.state[:2, 0])
    
    def _generate_final_path(self, goal_node):
        """
        Generate final path by backtracking from goal to start.
        Returns (waypoints, controls) where controls[i] is the control to apply at waypoints[i]
        to reach waypoints[i+1].
        """
        waypoints = []
        controls = []
        node = goal_node
        
        # Backtrack from goal to start
        while node is not None:
            # Extract position
            if self.model == 'SingleIntegrator2D' or self.model == 'SingleIntegrator2DOpenLoop':
                waypoints.append([node.state[0, 0], node.state[1, 0]])
            elif self.model == 'DoubleIntegrator2D':
                waypoints.append([node.state[0, 0], node.state[1, 0]])
            else:
                waypoints.append(node.state[:2, 0].tolist())
            
            # Extract control (control stored in node is the control applied at parent to reach this node)
            if node.control is not None:
                controls.append(node.control.flatten().tolist())
            
            node = node.parent
        
        # Reverse to get start-to-goal order
        waypoints.reverse()
        controls.reverse()
        
        # Now controls[i] is the control that was applied at waypoints[i] to reach waypoints[i+1]
        # But we need to shift: controls should have one less element than waypoints
        # (no control needed at the last waypoint)
        
        return np.array(waypoints[1:]), np.array(controls)
    
    def _draw_graph(self, rnd_node=None):
        """Draw RRT tree"""
        if self.ax is None or not self.show_animation:
            return
        
        plt.figure(self.fig.number)
        
        # Draw tree edges
        for node in self.node_list:
            if node.parent is not None:
                self.ax.plot([node.state[0, 0], node.parent.state[0, 0]],
                           [node.state[1, 0], node.parent.state[1, 0]],
                           "-g", alpha=0.3, linewidth=0.5)
        
        # Draw random node
        if rnd_node is not None:
            self.ax.plot(rnd_node.state[0, 0], rnd_node.state[1, 0], "^k", markersize=4)
        
        plt.pause(0.01)
    
    def _draw_final_path(self, path):
        """Draw final path"""
        if self.ax is None or path is None or not self.show_animation:
            return
        
        plt.figure(self.fig.number)
        self.ax.plot(path[:, 0], path[:, 1], '-r', linewidth=3, label='RRT Path')
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)


   


    


