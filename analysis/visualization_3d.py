"""
3D Visualization module for NeuroMechFly embodied simulation.
Provides matplotlib-based 3D rendering similar to MuJoCo viewer but in Python.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ArenaVisualizer3D:
    """
    3D visualizer for fly behavior in arena.
    Renders trajectory, odor gradient, and fly orientation.
    """
    
    def __init__(self, arena_config: Dict):
        """
        Initialize the 3D visualizer.
        
        Args:
            arena_config: Configuration with arena dimensions
        """
        self.width = arena_config.get('arena', {}).get('width', 100.0)
        self.height = arena_config.get('arena', {}).get('height', 100.0)
        self.depth = arena_config.get('arena', {}).get('depth', 50.0)
        self.food_pos = arena_config.get('odor', {}).get('food_position', [50, 50, 0])
    
    def plot_3d_trajectory(self, logged_data: Dict, output_file: Optional[str] = None,
                          with_odor_field: bool = True):
        """
        Create interactive 3D plot of fly trajectory.
        
        Args:
            logged_data: Dictionary with logged simulation data
            output_file: File to save figure
            with_odor_field: Include odor gradient visualization
        """
        positions = np.array(logged_data.get('position', []))
        odors = np.array(logged_data.get('odor_input', []))
        
        if positions.size == 0:
            logger.warning("No position data for 3D visualization")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory with color gradient based on odor
        if odors.size == len(positions):
            # Color-coded by odor concentration
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                c=odors, cmap='YlGn', s=1, alpha=0.6, label='Trajectory')
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Odor Concentration')
        else:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                   'b-', alpha=0.5, linewidth=0.5, label='Trajectory')
        
        # Start and end points
        ax.scatter(*positions[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(*positions[-1], color='red', s=100, marker='s', label='End', zorder=5)
        
        # Food source
        ax.scatter(*self.food_pos, color='gold', s=300, marker='*', 
                  label='Food', edgecolors='orange', linewidth=2, zorder=10)
        
        # Arena boundaries
        self._draw_arena_boundary(ax)
        
        # Odor field as wire mesh (heatmap)
        if with_odor_field:
            self._add_odor_field(ax, logged_data)
        
        # Labels and styling
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Fly Trajectory in Arena - NeuroMechFly Embodied Simulation')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, max(self.depth, np.max(positions[:, 2]) + 5))
        
        # Adjust viewing angle
        ax.view_init(elev=25, azim=45)
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            logger.info(f"3D trajectory plot saved to {output_file}")
        
        return fig, ax
    
    def _draw_arena_boundary(self, ax):
        """Draw arena boundaries as wireframe."""
        # Arena corners
        corners = np.array([
            [0, 0, 0], [self.width, 0, 0], [self.width, self.height, 0], [0, self.height, 0],
            [0, 0, self.depth], [self.width, 0, self.depth], 
            [self.width, self.height, self.depth], [0, self.height, self.depth]
        ])
        
        # Draw edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]
        
        for edge in edges:
            points = corners[list(edge)]
            ax.plot3D(*points.T, 'k-', alpha=0.2, linewidth=1)
    
    def _add_odor_field(self, ax, logged_data: Dict):
        """Add semi-transparent odor field visualization."""
        try:
            # Create grid for odor field
            x = np.linspace(0, self.width, 15)
            y = np.linspace(0, self.height, 15)
            X, Y = np.meshgrid(x, y)
            
            # Calculate odor at each point (Gaussian around food)
            sigma = self.width / 4.0
            food_x, food_y = self.food_pos[0], self.food_pos[1]
            
            Z = np.exp(-((X - food_x)**2 + (Y - food_y)**2) / (2 * sigma**2))
            
            # Plot surface translucently
            ax.plot_surface(X, Y, Z * self.depth * 0.3, alpha=0.15, cmap='Greens', 
                           linewidth=0, antialiased=True)
        except Exception as e:
            logger.debug(f"Could not plot odor field: {e}")
    
    def plot_neural_heatmap(self, logged_data: Dict, output_file: Optional[str] = None):
        """
        Create heatmap of neural activity over time.
        
        Args:
            logged_data: Dictionary with logged simulation data
            output_file: File to save figure
        """
        try:
            spikes = logged_data.get('neural_spikes', [])
            if not spikes:
                return
            
            # Extract spike counts
            layers = ['orn', 'pn', 'kc', 'mbon', 'dn']
            spike_data = {layer: [] for layer in layers}
            
            for spike_dict in spikes:
                if spike_dict:
                    for layer in layers:
                        if layer in spike_dict:
                            spike_data[layer].append(np.sum(spike_dict[layer]))
                        else:
                            spike_data[layer].append(0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            
            heatmap_data = np.array([spike_data[layer] for layer in layers])
            
            im = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([l.upper() for l in layers])
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Neural Layer')
            ax.set_title('Neural Activity Heatmap (Spike Counts per Layer)')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Spike Count')
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"Neural heatmap saved to {output_file}")
            
            return fig, ax
        except Exception as e:
            logger.warning(f"Could not create neural heatmap: {e}")
    
    def plot_behavior_analysis(self, logged_data: Dict, output_file: Optional[str] = None):
        """
        Create multi-panel analysis plot showing:
        - Velocity profile
        - Turning angle
        - Odor response
        - Motor commands
        """
        try:
            positions = np.array(logged_data.get('position', []))
            odors = np.array(logged_data.get('odor_input', []))
            times = np.array(logged_data.get('time', []))
            
            if positions.size < 2:
                return
            
            # Calculate velocity and angular changes
            deltas = np.diff(positions, axis=0)
            velocities = np.linalg.norm(deltas, axis=1)
            
            # Calculate orientation (heading direction)
            orientations = np.arctan2(deltas[:, 1], deltas[:, 0])
            angular_velocity = np.diff(orientations)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Velocity over time
            ax = axes[0, 0]
            ax.plot(times[1:], velocities, 'b-', linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Velocity (mm/s)')
            ax.set_title('Fly Velocity')
            ax.grid(True, alpha=0.3)
            
            # Angular velocity (turning)
            ax = axes[0, 1]
            ax.plot(times[2:], angular_velocity * 180 / np.pi, 'r-', linewidth=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angular Velocity (deg/s)')
            ax.set_title('Turning Rate')
            ax.grid(True, alpha=0.3)
            
            # Odor input
            ax = axes[1, 0]
            ax.fill_between(times, 0, odors, alpha=0.5, color='green')
            ax.plot(times, odors, 'g-', linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Odor Concentration')
            ax.set_title('Olfactory Input')
            ax.grid(True, alpha=0.3)
            
            # Distance from food
            ax = axes[1, 1]
            food_pos = np.array(self.food_pos[:2])
            distances = np.linalg.norm(positions[:, :2] - food_pos, axis=1)
            ax.plot(times, distances, 'orange', linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Distance to Food (mm)')
            ax.set_title('Approach to Food Source')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"Behavior analysis plot saved to {output_file}")
            
            return fig, axes
        except Exception as e:
            logger.warning(f"Could not create behavior analysis: {e}")
    
    def create_interactive_report(self, logged_data: Dict, output_dir: str):
        """
        Generate a comprehensive visual report with multiple plots.
        
        Args:
            logged_data: Dictionary with logged simulation data
            output_dir: Directory to save all plots
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating comprehensive visual report to {output_dir}")
        
        # 3D Trajectory
        try:
            fig, _ = self.plot_3d_trajectory(
                logged_data,
                output_file=str(output_path / '1_trajectory_3d.png'),
                with_odor_field=True
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create 3D trajectory: {e}")
        
        # Neural heatmap
        try:
            fig, _ = self.plot_neural_heatmap(
                logged_data,
                output_file=str(output_path / '2_neural_heatmap.png')
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create neural heatmap: {e}")
        
        # Behavior analysis
        try:
            fig, _ = self.plot_behavior_analysis(
                logged_data,
                output_file=str(output_path / '3_behavior_analysis.png')
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to create behavior analysis: {e}")
        
        logger.info("Visual report generation complete")
