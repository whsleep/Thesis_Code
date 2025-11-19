import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.spatial.distance import cdist

class SpatiotemporalSamplingVisualizer:
    def __init__(self, A=2.0, N=7, T=3.0, K=20, obstacles=None):
        """Initialize Spatiotemporal Sampling Visualizer
        
        Parameters:
            A: Maximum absolute acceleration
            N: Number of sampling points per dimension
            T: Prediction time horizon (seconds)
            K: Number of time steps
            obstacles: List of obstacles, each as [x, y, radius] or [x, y, width, height]
        """
        self.A = A
        self.N = N
        self.T = T
        self.K = K
        self.dt = T / K
        
        # Generate control input set
        self.accels = np.linspace(-A, A, N)
        self.control_set = np.array(np.meshgrid(self.accels, self.accels)).T.reshape(-1, 2)
        
        # Time sequence
        self.time_points = np.linspace(0, T, K)
        
        # Initial state [x, y, vx, vy]
        self.init_state = np.array([0, 0, 1.0, 0.5])
        
        # Obstacles
        self.obstacles = obstacles if obstacles else []
        
        # Color mapping
        self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.control_set)))
        
        # Set font for Chinese characters (if needed)
        plt.rcParams['font.family'] = ['DejaVu Sans']  # Use a font that supports English
        plt.rcParams['axes.unicode_minus'] = False
    
    def dynamics_model(self, s0, ax, ay, t):
        """Uniform acceleration motion model"""
        x0, y0, vx0, vy0 = s0
        x = x0 + vx0 * t + 0.5 * ax * t**2
        y = y0 + vy0 * t + 0.5 * ay * t**2
        vx = vx0 + ax * t
        vy = vy0 + ay * t
        return np.array([x, y, vx, vy])
    
    def generate_trajectories(self):
        """Generate all sampled trajectories"""
        trajectories = []
        for i, (ax_val, ay_val) in enumerate(self.control_set):
            trajectory = []
            for t in self.time_points:
                state = self.dynamics_model(self.init_state, ax_val, ay_val, t)
                trajectory.append(state)
            trajectories.append(np.array(trajectory))
        return trajectories
    
    def calculate_trajectory_cost(self, trajectory, target_point=None):
        """Calculate trajectory cost (distance to target + smoothness + obstacle penalty)"""
        if target_point is None:
            target_point = np.array([5, 3])  # Default target point
        
        # Target distance cost
        final_pos = trajectory[-1, :2]
        distance_cost = np.linalg.norm(final_pos - target_point)
        
        # Smoothness cost (acceleration variation)
        smoothness_cost = 0
        positions = trajectory[:, :2]
        if len(positions) > 2:
            accelerations = np.diff(np.diff(positions, axis=0), axis=0) / (self.dt**2)
            smoothness_cost = np.mean(np.linalg.norm(accelerations, axis=1))
        
        # Obstacle collision cost
        obstacle_cost = 0
        for pos in positions:
            for obstacle in self.obstacles:
                if len(obstacle) == 3:  # Circular obstacle
                    obs_pos, radius = obstacle[:2], obstacle[2]
                    dist = np.linalg.norm(pos - obs_pos)
                    if dist < radius:
                        obstacle_cost += 100 * (radius - dist)
                else:  # Rectangular obstacle
                    x, y, w, h = obstacle
                    if (x <= pos[0] <= x + w) and (y <= pos[1] <= y + h):
                        obstacle_cost += 100
        
        total_cost = distance_cost + 0.1 * smoothness_cost + obstacle_cost
        return total_cost
    
    def plot_static_visualization(self, target_point=None):
        """Plot static trajectory visualization"""
        if target_point is None:
            target_point = np.array([5, 3])
        
        trajectories = self.generate_trajectories()
        costs = [self.calculate_trajectory_cost(traj, target_point) for traj in trajectories]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spatiotemporal Sampling Visualization Analysis', fontsize=16, fontweight='bold')
        
        # 1. Trajectory spatial distribution
        ax1 = axes[0, 0]
        for i, trajectory in enumerate(trajectories):
            positions = trajectory[:, :2]
            ax1.plot(positions[:, 0], positions[:, 1], 
                    color=self.colors[i], alpha=0.7, linewidth=2,
                    label=f'a=({self.control_set[i, 0]:.1f},{self.control_set[i, 1]:.1f})')
        
        # Mark start and target points
        ax1.scatter(self.init_state[0], self.init_state[1], color='green', 
                   s=100, zorder=5, label='Start Point')
        ax1.scatter(target_point[0], target_point[1], color='red', 
                   s=100, zorder=5, label='Target Point')
        
        # Draw obstacles
        for obstacle in self.obstacles:
            if len(obstacle) == 3:
                circle = Circle(obstacle[:2], obstacle[2], color='red', alpha=0.3)
                ax1.add_patch(circle)
                ax1.text(obstacle[0], obstacle[1], 'Obstacle', 
                        ha='center', va='center', fontsize=10)
            else:
                rect = Rectangle(obstacle[:2], obstacle[2], obstacle[3], 
                               color='red', alpha=0.3)
                ax1.add_patch(rect)
                ax1.text(obstacle[0] + obstacle[2]/2, obstacle[1] + obstacle[3]/2, 
                        'Obstacle', ha='center', va='center', fontsize=10)
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('Spatial Distribution of Sampled Trajectories')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.axis('equal')
        
        # 2. Control input space heatmap
        ax2 = axes[0, 1]
        cost_matrix = np.zeros((self.N, self.N))
        for i, (ax_val, ay_val) in enumerate(self.control_set):
            idx_x = np.where(np.isclose(self.accels, ax_val))[0][0]
            idx_y = np.where(np.isclose(self.accels, ay_val))[0][0]
            cost_matrix[idx_y, idx_x] = costs[i]  # Note index order
        
        im = ax2.imshow(cost_matrix, cmap='hot', 
                       extent=[-self.A, self.A, -self.A, self.A],
                       origin='lower', aspect='auto')
        ax2.set_xlabel('a_x Acceleration')
        ax2.set_ylabel('a_y Acceleration')
        ax2.set_title('Cost Heatmap in Control Input Space')
        plt.colorbar(im, ax=ax2, label='Trajectory Cost')
        
        # Mark optimal control input
        min_cost_idx = np.argmin(costs)
        best_control = self.control_set[min_cost_idx]
        ax2.scatter(best_control[0], best_control[1], color='blue', s=100, 
                   marker='*', label='Optimal Control')
        ax2.legend()
        
        # 3. Time-position relationship
        ax3 = axes[1, 0]
        for i, trajectory in enumerate(trajectories):
            ax3.plot(self.time_points, trajectory[:, 0], 
                    color=self.colors[i], alpha=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('X Position')
        ax3.set_title('X Position Over Time')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost distribution histogram
        ax4 = axes[1, 1]
        ax4.hist(costs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.min(costs), color='red', linestyle='--', 
                   label=f'Min Cost: {np.min(costs):.2f}')
        ax4.set_xlabel('Trajectory Cost')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Trajectory Cost Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, target_point=None):
        """Create trajectory generation animation"""
        if target_point is None:
            target_point = np.array([5, 3])
        
        trajectories = self.generate_trajectories()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            current_time = frame * self.dt
            
            # Draw generated trajectory segments
            for i, trajectory in enumerate(trajectories):
                time_indices = self.time_points <= current_time
                if np.sum(time_indices) > 1:
                    visible_trajectory = trajectory[time_indices]
                    positions = visible_trajectory[:, :2]
                    
                    # Use LineCollection for color gradient
                    points = positions.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, self.T)
                    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
                    lc.set_array(self.time_points[time_indices][:-1])
                    ax.add_collection(lc)
            
            # Draw obstacles
            for obstacle in self.obstacles:
                if len(obstacle) == 3:
                    circle = Circle(obstacle[:2], obstacle[2], color='red', alpha=0.3)
                    ax.add_patch(circle)
                else:
                    rect = Rectangle(obstacle[:2], obstacle[2], obstacle[3], 
                                   color='red', alpha=0.3)
                    ax.add_patch(rect)
            
            # Mark start and target points
            ax.scatter(self.init_state[0], self.init_state[1], color='green', 
                      s=100, zorder=5, label='Start Point')
            ax.scatter(target_point[0], target_point[1], color='red', 
                      s=100, zorder=5, label='Target Point')
            
            ax.set_xlim(-2, 8)
            ax.set_ylim(-2, 6)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title(f'Spatiotemporal Sampling Process (Time: {current_time:.1f}s)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect('equal')
            
            return ax,
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.time_points),
                                      interval=200, blit=False, repeat=True)
        plt.close()
        return anim
    
    def create_interactive_3d_plot(self):
        """Create interactive 3D spatiotemporal visualization"""
        trajectories = self.generate_trajectories()
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
            subplot_titles=('3D Spatiotemporal Trajectory Visualization', 'Control Input Space')
        )
        
        # 3D trajectory plot
        for i, trajectory in enumerate(trajectories):
            positions = trajectory[:, :2]
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=self.time_points,
                mode='lines',
                line=dict(width=6, color=f'rgba({int(self.colors[i][0]*255)},{int(self.colors[i][1]*255)},{int(self.colors[i][2]*255)},0.8)'),
                name=f'a=({self.control_set[i, 0]:.1f},{self.control_set[i, 1]:.1f})',
                hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>t: %{z:.2f}s'
            ), row=1, col=1)
        
        # Control input space
        control_x, control_y = zip(*self.control_set)
        fig.add_trace(go.Scatter(
            x=control_x, y=control_y,
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Control Sampling Points',
            hovertemplate='a_x: %{x:.2f}<br>a_y: %{y:.2f}'
        ), row=1, col=2)
        
        fig.update_layout(
            title_text="Interactive Spatiotemporal Sampling Visualization",
            scene=dict(
                xaxis_title='X Position',
                yaxis_title='Y Position',
                zaxis_title='Time (s)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=500
        )
        
        return fig
    
    def find_optimal_trajectory(self, target_point=None):
        """Find optimal trajectory"""
        if target_point is None:
            target_point = np.array([5, 3])
        
        trajectories = self.generate_trajectories()
        costs = [self.calculate_trajectory_cost(traj, target_point) for traj in trajectories]
        
        min_cost_idx = np.argmin(costs)
        optimal_trajectory = trajectories[min_cost_idx]
        optimal_control = self.control_set[min_cost_idx]
        
        print(f"Optimal control input: a_x={optimal_control[0]:.2f}, a_y={optimal_control[1]:.2f}")
        print(f"Minimum trajectory cost: {costs[min_cost_idx]:.2f}")
        print(f"Final position: ({optimal_trajectory[-1, 0]:.2f}, {optimal_trajectory[-1, 1]:.2f})")
        
        return optimal_trajectory, optimal_control, min_cost_idx

def main():
    """Main function example"""
    # Set obstacles
    obstacles = [
        [2, 1, 0.5],  # Circular obstacle [x, y, radius]
        [3, 3, 1, 0.5]  # Rectangular obstacle [x, y, width, height]
    ]
    
    # Create visualizer
    visualizer = SpatiotemporalSamplingVisualizer(
        A=2.0, N=7, T=4.0, K=25, obstacles=obstacles
    )
    
    # Target point
    target_point = np.array([6, 4])
    
    print("=== Spatiotemporal Sampling Visualization System ===")
    print(f"Number of control samples: {visualizer.N}×{visualizer.N} = {len(visualizer.control_set)}")
    print(f"Prediction horizon: {visualizer.T} seconds, Time steps: {visualizer.K}")
    print(f"Target point position: ({target_point[0]}, {target_point[1]})")
    
    # 1. Find optimal trajectory
    print("\n1. Optimal trajectory analysis:")
    optimal_traj, optimal_control, opt_idx = visualizer.find_optimal_trajectory(target_point)
    
    # 2. Generate static visualization
    print("\n2. Generating static visualization...")
    fig_static = visualizer.plot_static_visualization(target_point)
    plt.show()
    
    # 3. Generate animation
    print("3. Generating trajectory sampling animation...")
    anim = visualizer.create_animation(target_point)
    
    # Save animation (optional)
    try:
        anim.save('spatiotemporal_sampling.gif', writer='pillow', fps=5, dpi=100)
        print("Animation saved as 'spatiotemporal_sampling.gif'")
    except Exception as e:
        print(f"Animation saving failed: {e}")
        print("Displaying animation preview...")
        plt.show()
    
    # 4. Generate interactive visualization
    print("4. Generating interactive 3D visualization...")
    fig_interactive = visualizer.create_interactive_3d_plot()
    fig_interactive.show()
    
    # 5. Generate trajectory data report
    print("\n5. Trajectory data statistics:")
    trajectories = visualizer.generate_trajectories()
    costs = [visualizer.calculate_trajectory_cost(traj, target_point) for traj in trajectories]
    
    stats_df = pd.DataFrame({
        'a_x': visualizer.control_set[:, 0],
        'a_y': visualizer.control_set[:, 1],
        'final_x': [traj[-1, 0] for traj in trajectories],
        'final_y': [traj[-1, 1] for traj in trajectories],
        'cost': costs
    })
    
    print(f"\nTrajectory statistics summary:")
    print(f"Average cost: {np.mean(costs):.2f} ± {np.std(costs):.2f}")
    print(f"Cost range: [{np.min(costs):.2f}, {np.max(costs):.2f}]")
    print(f"Number of trajectories successfully reaching near target: {sum(1 for c in costs if c < 10)}")
    
    return visualizer, stats_df

if __name__ == "__main__":
    # Run main program
    visualizer, stats_df = main()
    
    # Optional: save statistical results
    stats_df.to_csv('trajectory_statistics.csv', index=False)
    print("\nTrajectory statistics saved as 'trajectory_statistics.csv'")