"""
Interactive 3D Real-Time Visualizer for NeuroMechFly Embodied Simulation.
Uses Vispy for fast OpenGL rendering similar to MuJoCo viewer.
"""

import numpy as np
from vispy import app, scene, geometry, visuals
from vispy.color import Color
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FlyVisualizer3D:
    """
    Real-time 3D interactive visualizer for fly simulation.
    Renders arena, odor field, fly body with skeleton, and trajectory.
    """
    
    def __init__(self, arena_config: Dict, window_size=(1200, 900)):
        """
        Initialize Vispy-based 3D visualizer.
        
        Args:
            arena_config: Arena configuration
            window_size: Window dimensions
        """
        self.arena_config = arena_config
        self.arena_params = arena_config.get('arena', {})
        self.odor_params = arena_config.get('odor', {})
        
        # Arena dimensions
        self.width = self.arena_params.get('width', 100.0)
        self.height = self.arena_params.get('height', 100.0)
        self.depth = self.arena_params.get('depth', 50.0)
        self.food_pos = np.array(self.odor_params.get('food_position', [50, 50, 0]))
        
        # Create canvas and 3D view
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=window_size,
            show=False,
            title='NeuroMechFly 3D Embodied Simulation'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up='z')
        self.view.camera.distance = min(self.width, self.height) * 2
        self.view.camera.center = [self.width/2, self.height/2, self.depth/2]
        
        # Setup rendering
        self._setup_scene()
        
        # Tracking
        self.trajectory_points = []
        self.trajectory_colors = []
        self.trajectory_line = None
        
        # Performance
        self.frame_count = 0
        self.update_interval = 1  # Update visualization every N steps
        
        logger.info(f"FlyVisualizer3D initialized: {window_size[0]}x{window_size[1]}")
    
    def _setup_scene(self):
        """Setup arena and environment visualization."""
        # Arena floor
        floor = scene.visuals.Box(
            width=self.width, height=self.height, depth=0.1,
            color=Color('#555555'),
            parent=self.view.scene
        )
        floor.transform.translate([self.width/2, self.height/2, -0.05])
        
        # Arena walls (wireframe)
        self._draw_arena_boundaries()
        
        # Food source marker
        food_marker = scene.visuals.Sphere(
            radius=2.0, color=Color('gold'),
            parent=self.view.scene
        )
        food_marker.transform.translate(self.food_pos)
        
        # Odor field visualization (semi-transparent)
        self._draw_odor_field()
        
        # Grid
        self.view.camera.set_range(x=(0, self.width), y=(0, self.height), z=(0, self.depth))
    
    def _draw_arena_boundaries(self):
        """Draw arena boundaries as lines."""
        corners = np.array([
            [0, 0, 0], [self.width, 0, 0], 
            [self.width, self.height, 0], [0, self.height, 0],
            [0, 0, self.depth], [self.width, 0, self.depth],
            [self.width, self.height, self.depth], [0, self.height, self.depth]
        ], dtype=np.float32)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical
        ]
        
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            line = scene.visuals.Line(
                pos=np.array([p1, p2]),
                color=Color('#CCCCCC'),
                width=2,
                antialias=True,
                parent=self.view.scene
            )
    
    def _draw_odor_field(self):
        """Draw gradient field as semi-transparent surface."""
        # Create grid
        x = np.linspace(0, self.width, 12)
        y = np.linspace(0, self.height, 12)
        X, Y = np.meshgrid(x, y)
        
        # Odor Gaussian
        sigma = self.width / 4.0
        Z = np.exp(-((X - self.food_pos[0])**2 + (Y - self.food_pos[1])**2) / (2 * sigma**2))
        Z = Z * self.depth * 0.3  # Scale to depth
        
        # Create surface
        vertices = np.zeros((X.shape[0] * X.shape[1], 3))
        vertices[:, 0] = X.ravel()
        vertices[:, 1] = Y.ravel()
        vertices[:, 2] = Z.ravel()
        
        # Simple mesh (no texture for now)
        # Could implement as wireframe
        points = scene.visuals.Markers(
            pos=vertices,
            size=2,
            color=Color((0.2, 0.8, 0.2, 0.3)),
            parent=self.view.scene
        )
    
    def render_fly(self, body_frame: Dict):
        """
        Render fly body in 3D.
        
        Args:
            body_frame: Body state dictionary from FlyBody.get_body_frame()
        """
        # COM
        com_pos = body_frame['com_pos']
        
        # HEAD
        head_pos = body_frame['head_pos']
        head_sphere = scene.visuals.Sphere(
            radius=0.7, color=Color('red'),
            parent=self.view.scene
        )
        head_sphere.transform.translate(head_pos)
        
        # THORAX (center)
        thorax_sphere = scene.visuals.Sphere(
            radius=0.5, color=Color('orange'),
            parent=self.view.scene
        )
        thorax_sphere.transform.translate(com_pos)
        
        # ABDOMEN
        abdomen_pos = body_frame['abdomen_pos']
        abdomen_sphere = scene.visuals.Sphere(
            radius=0.6, color=Color('brown'),
            parent=self.view.scene
        )
        abdomen_sphere.transform.translate(abdomen_pos)
        
        # BODY AXIS (head to abdomen)
        body_axis = scene.visuals.Line(
            pos=np.array([head_pos, abdomen_pos]),
            color=Color('white'),
            width=2,
            parent=self.view.scene
        )
        
        # LEGS (sticks)
        legs = body_frame['legs']  # (6, 3, 3)
        leg_colors = [
            Color('green'),   # Front-L
            Color('lime'),    # Front-R
            Color('cyan'),    # Middle-L
            Color('blue'),    # Middle-R
            Color('magenta'), # Hind-L
            Color('violet'),  # Hind-R
        ]
        
        for leg_id in range(6):
            leg_points = legs[leg_id]  # (3, 3)
            
            # Draw leg segments
            for segment in range(len(leg_points) - 1):
                p1 = leg_points[segment]
                p2 = leg_points[segment + 1]
                
                line = scene.visuals.Line(
                    pos=np.array([p1, p2]),
                    color=leg_colors[leg_id],
                    width=2,
                    parent=self.view.scene
                )
            
            # Draw leg end effector (claw)
            claw = scene.visuals.Sphere(
                radius=0.2, color=leg_colors[leg_id],
                parent=self.view.scene
            )
            claw.transform.translate(leg_points[-1])
        
        # WINGS
        # Simple triangle wings
        wing_size = 3.0
        heading = body_frame['heading']
        right_vec = np.array([-heading[1], heading[0], 0])  # Perpendicular
        
        # Left wing
        wing_left_base = com_pos + heading * 0.5 - right_vec * 0.3
        wing_left_tip = wing_left_base + right_vec * wing_size
        
        wing_left_line = scene.visuals.Line(
            pos=np.array([com_pos, wing_left_tip]),
            color=Color((0.5, 0.5, 0.8, 0.7)),
            width=3,
            parent=self.view.scene
        )
        
        # Right wing
        wing_right_base = com_pos + heading * 0.5 + right_vec * 0.3
        wing_right_tip = wing_right_base - right_vec * wing_size
        
        wing_right_line = scene.visuals.Line(
            pos=np.array([com_pos, wing_right_tip]),
            color=Color((0.5, 0.5, 0.8, 0.7)),
            width=3,
            parent=self.view.scene
        )
    
    def add_trajectory_point(self, position: np.ndarray, odor: float):
        """
        Add point to trajectory trace.
        
        Args:
            position: 3D position
            odor: Odor concentration (for coloring)
        """
        self.trajectory_points.append(position.copy())
        
        # Color by odor concentration (green for high)
        intensity = np.clip(odor, 0, 1)
        color = np.array([1 - intensity, intensity, 0.3, 0.5])
        self.trajectory_colors.append(color)
        
        # Keep only last 1000 points
        if len(self.trajectory_points) > 1000:
            self.trajectory_points.pop(0)
            self.trajectory_colors.pop(0)
    
    def update_trajectory_display(self):
        """Update trajectory line visualization."""
        if len(self.trajectory_points) > 1:
            # Remove old trajectory
            if self.trajectory_line is not None:
                self.trajectory_line.parent = None
            
            # Draw new trajectory
            traj_array = np.array(self.trajectory_points, dtype=np.float32)
            self.trajectory_line = scene.visuals.Line(
                pos=traj_array,
                color=np.array(self.trajectory_colors, dtype=np.float32),
                width=1,
                connect='segments',
                antialias=True,
                parent=self.view.scene
            )
    
    def show(self):
        """Display the visualization."""
        self.canvas.show()
    
    def update(self):
        """Update visualization (call in main loop)."""
        # Vispy handles rendering automatically
        pass
    
    def close(self):
        """Close visualization window."""
        self.canvas.close()


# Standalone viewer for saved data
def view_3d_trajectory(sim_result_path: str):
    """
    Load and view 3D trajectory from simulation results.
    
    Args:
        sim_result_path: Path to HDF5 simulation data
    """
    import h5py
    
    with h5py.File(sim_result_path, 'r') as f:
        positions = np.array(f['position'])
        odors = np.array(f['odor_input'])
    
    # Setup visualizer (dummy config)
    config = {
        'arena': {'width': 100, 'height': 100, 'depth': 50},
        'odor': {'food_position': [50, 50, 0]}
    }
    
    viz = FlyVisualizer3D(config)
    
    # Add trajectory
    for pos, odor in zip(positions, odors):
        viz.add_trajectory_point(pos, odor)
    
    viz.update_trajectory_display()
    viz.show()
