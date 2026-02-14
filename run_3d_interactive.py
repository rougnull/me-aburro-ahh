#!/usr/bin/env python3
"""
Interactive 3D Real-Time NeuroMechFly Simulation Viewer.
Displays live neural-controlled fly movement in 3D space.
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque
import threading
import time
import logging

try:
    from vispy import app, scene, visuals
    from vispy.color import Color
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False
    print("Warning: Vispy not available, will use matplotlib fallback")

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.simulation import NeuroMechFlySimulation
from core.environment import Arena
from brain.olfactory_circuit import OlfactoryCircuit
from body.realistic_body import FlyBody, RealisticFlyInterface
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeVispy3DViewer:
    """Real-time Vispy 3D viewer for embodied simulation."""
    
    def __init__(self, config: dict, max_trail_points=500):
        """
        Initialize interactive 3D viewer.
        
        Args:
            config: Configuration dictionary
            max_trail_points: Maximum trajectory points to display
        """
        self.config = config
        self.max_trail_points = max_trail_points
        
        # Arena params
        arena_cfg = config.get('arena', {})
        self.arena_width = arena_cfg.get('width', 100)
        self.arena_height = arena_cfg.get('height', 100)
        self.arena_depth = arena_cfg.get('depth', 50)
        
        odor_cfg = config.get('odor', {})
        self.food_pos = np.array(odor_cfg.get('food_position', [50, 50, 0]))
        
        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1400, 900),
            show=False,
            title='NeuroMechFly 3D Embodied Simulation - Real-Time',
            bgcolor=Color('#1a1a1a')
        )
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(up='z')
        self.view.camera.distance = max(self.arena_width, self.arena_height) * 1.8
        self.view.camera.center = [self.arena_width/2, self.arena_height/2, self.arena_depth/2]
        
        # Setup scene
        self._setup_arena()
        
        # Visual objects
        self.fly_head = None
        self.fly_thorax = None
        self.fly_abdomen = None
        self.fly_body_line = None
        self.fly_legs = []
        self.fly_wings = []
        
        # Trail
        self.trail_points = deque(maxlen=max_trail_points)
        self.trail_line = None
        self.trail_colors = deque(maxlen=max_trail_points)
        
        # Text/stats
        self.text_info = scene.visuals.Text(
            text='Initializing...',
            pos=(10, 10),
            font_size=12,
            color='white',
            parent=self.canvas.scene
        )
        
        logger.info("Realtime Vispy3D viewer initialized")
    
    def _setup_arena(self):
        """Setup arena visualization."""
        # Floor
        floor = scene.visuals.Box(
            width=self.arena_width,
            height=self.arena_height,
            depth=0.5,
            color=Color('#333333'),
            parent=self.view.scene
        )
        floor.transform.translate((self.arena_width/2, self.arena_height/2, -0.25))
        
        # Walls (wireframe)
        corners = [
            [0, 0, 0], [self.arena_width, 0, 0],
            [self.arena_width, self.arena_height, 0], [0, self.arena_height, 0],
            [0, 0, self.arena_depth], [self.arena_width, 0, self.arena_depth],
            [self.arena_width, self.arena_height, self.arena_depth], 
            [0, self.arena_height, self.arena_depth]
        ]
        corners = np.array(corners, dtype=np.float32)
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        for e in edges:
            line = scene.visuals.Line(
                pos=np.array([corners[e[0]], corners[e[1]]]),
                color=Color('#555555'),
                width=1.5,
                antialias=True,
                parent=self.view.scene
            )
        
        # Food source
        food_sphere = scene.visuals.Sphere(
            radius=2,
            color=Color('gold'),
            parent=self.view.scene,
            shading='smooth'
        )
        food_sphere.transform.translate(tuple(self.food_pos))
        
        # Odor field (gradient)
        self._visualize_odor_field()
    
    def _visualize_odor_field(self):
        """Add odor gradient visualization."""
        x = np.linspace(0, self.arena_width, 15)
        y = np.linspace(0, self.arena_height, 15)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian
        sigma = self.arena_width / 4
        Z = np.exp(-((X - self.food_pos[0])**2 + (Y - self.food_pos[1])**2) / (2 * sigma**2))
        Z = Z * self.arena_depth * 0.25
        
        # Create vertices
        verts = np.zeros((X.size, 3), dtype=np.float32)
        verts[:, 0] = X.ravel()
        verts[:, 1] = Y.ravel()
        verts[:, 2] = Z.ravel()
        
        # Color by intensity
        colors = np.zeros((X.size, 4), dtype=np.float32)
        odor_vals = Z.ravel() / (self.arena_depth * 0.25)
        colors[:, 1] = odor_vals  # Green channel
        colors[:, 0] = 1 - odor_vals  # Red channel (inverse)
        colors[:, 3] = 0.15  # Alpha (transparent)
        
        markers = scene.visuals.Markers(
            pos=verts,
            size=3,
            face_color=colors,
            parent=self.view.scene
        )
    
    def update_fly(self, body, position, heading):
        """
        Update fly visualization.
        
        Args:
            body: FlyBody object
            position: 3D position
            heading: Heading vector
        """
        # Remove old fly parts
        for obj in [self.fly_head, self.fly_thorax, self.fly_abdomen, self.fly_body_line]:
            if obj is not None:
                obj.parent = None
        
        for leg in self.fly_legs:
            leg.parent = None
        self.fly_legs = []
        
        for wing in self.fly_wings:
            wing.parent = None
        self.fly_wings = []
        
        # Get body frame
        frame = body.get_body_frame()
        
        # HEAD (red)
        self.fly_head = scene.visuals.Sphere(
            radius=0.8,
            color=Color('red'),
            parent=self.view.scene,
            shading='smooth'
        )
        self.fly_head.transform.translate(tuple(frame['head_pos']))
        
        # THORAX (orange)
        self.fly_thorax = scene.visuals.Sphere(
            radius=0.6,
            color=Color('orange'),
            parent=self.view.scene,
            shading='smooth'
        )
        self.fly_thorax.transform.translate(tuple(position))
        
        # ABDOMEN (brown)
        self.fly_abdomen = scene.visuals.Sphere(
            radius=0.7,
            color=Color('brown'),
            parent=self.view.scene,
            shading='smooth'
        )
        self.fly_abdomen.transform.translate(tuple(frame['abdomen_pos']))
        
        # BODY AXIS
        self.fly_body_line = scene.visuals.Line(
            pos=np.array([frame['head_pos'], frame['abdomen_pos']]),
            color=Color('white'),
            width=2,
            parent=self.view.scene
        )
        
        # LEGS
        leg_colors = [
            Color('lime'), Color('lime'),
            Color('cyan'), Color('cyan'),
            Color('magenta'), Color('magenta'),
        ]
        
        legs = frame['legs']
        for leg_id in range(6):
            leg_pts = legs[leg_id]
            
            # Leg segments (lines)
            for seg in range(len(leg_pts) - 1):
                line = scene.visuals.Line(
                    pos=np.array([leg_pts[seg], leg_pts[seg + 1]]),
                    color=leg_colors[leg_id],
                    width=1,
                    parent=self.view.scene
                )
                self.fly_legs.append(line)
            
            # Claw (small sphere)
            claw = scene.visuals.Sphere(
                radius=0.15,
                color=leg_colors[leg_id],
                parent=self.view.scene,
                shading='smooth'
            )
            claw.transform.translate(tuple(leg_pts[-1]))
            self.fly_legs.append(claw)
        
        # WINGS
        wing_size = 3.0
        right_vec = np.array([-heading[1], heading[0], 0])
        
        # Left wing
        wing_left_base = position + heading * 0.5 - right_vec * 0.3
        wing_left_tip = wing_left_base + right_vec * wing_size
        wing_l = scene.visuals.Line(
            pos=np.array([position, wing_left_tip]),
            color=Color((0.3, 0.6, 1, 0.7)),
            width=2,
            parent=self.view.scene
        )
        self.fly_wings.append(wing_l)
        
        # Right wing
        wing_right_base = position + heading * 0.5 + right_vec * 0.3
        wing_right_tip = wing_right_base - right_vec * wing_size
        wing_r = scene.visuals.Line(
            pos=np.array([position, wing_right_tip]),
            color=Color((0.3, 0.6, 1, 0.7)),
            width=2,
            parent=self.view.scene
        )
        self.fly_wings.append(wing_r)
    
    def add_trail_point(self, position, odor_intensity):
        """Add point to trajectory trail."""
        self.trail_points.append(position.copy())
        
        # Color by odor (red → yellow → green)
        intensity = np.clip(odor_intensity, 0, 1)
        if intensity < 0.5:
            color = (1, intensity * 2, 0, 0.6)
        else:
            color = (1 - (intensity - 0.5) * 2, 1, 0, 0.6)
        
        self.trail_colors.append(color)
    
    def update_trail_display(self):
        """Update trail visualization."""
        if self.trail_line is not None:
            self.trail_line.parent = None
        
        if len(self.trail_points) > 1:
            pts = np.array(list(self.trail_points), dtype=np.float32)
            cols = np.array(list(self.trail_colors), dtype=np.float32)
            
            self.trail_line = scene.visuals.Line(
                pos=pts,
                color=cols,
                width=1.5,
                connect='segments',
                antialias=True,
                parent=self.view.scene
            )
    
    def update_stats(self, step, position, velocity, odor):
        """Update on-screen statistics."""
        text = f"Step: {step:6d} | Pos: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}) mm | "
        text += f"Vel: {np.linalg.norm(velocity):.3f} mm/s | Odor: {odor:.3f}"
        self.text_info.text = text
    
    def show(self):
        """Display viewer."""
        self.canvas.show()
    
    def update(self):
        """Process events."""
        pass
    
    def close(self):
        """Close viewer."""
        self.canvas.close()


def run_interactive_simulation(duration=30):
    """Run interactive 3D simulation."""
    
    # Load config
    config_dir = project_root / 'config'
    with open(config_dir / 'environment.yaml') as f:
        env_cfg = yaml.safe_load(f) or {}
    with open(config_dir / 'fly_params.yaml') as f:
        fly_cfg = yaml.safe_load(f) or {}
    with open(config_dir / 'brain_params.yaml') as f:
        brain_cfg = yaml.safe_load(f) or {}
    
    config = {}
    config.update(env_cfg)
    config.update(brain_cfg)
    config['fly_params'] = fly_cfg
    
    # Initialize components
    logger.info("Initializing embodied simulation...")
    
    arena = Arena(
        width=config['arena']['width'],
        height=config['arena']['height'],
        depth=config['arena']['depth'],
        food_position=config['odor']['food_position'],
        food_intensity=config['odor'].get('food_intensity', 1.0),
        diffusion_coeff=config['odor'].get('diffusion_coefficient', 0.1),
        decay_rate=config['odor'].get('decay_rate', 0.05)
    )
    
    brain = OlfactoryCircuit(config)
    fly_body = FlyBody()
    fly_interface = RealisticFlyInterface(fly_body)
    
    sim = NeuroMechFlySimulation(fly_interface, brain, arena, config)
    
    # Initialize viewer
    if not VISPY_AVAILABLE:
        logger.error("Vispy is required for interactive visualization")
        return
    
    viewer = RealtimeVispy3DViewer(config)
    viewer.show()
    
    # Run simulation
    n_steps = int(duration * 1000)  # ms timesteps
    logger.info(f"Running {n_steps} steps ({duration}s)...")
    
    for step in range(n_steps):
        # Step simulation
        sim.step()
        pos = fly_interface.get_body_position()
        heading = fly_interface.get_orientation()
        vel = fly_interface.get_velocity()
        odor = arena.get_odor_concentration(pos)
        
        # Update visualization every 10 steps
        if step % 10 == 0:
            viewer.update_fly(fly_body, pos, heading)
            viewer.add_trail_point(pos, odor)
            viewer.update_trail_display()
            viewer.update_stats(step, pos, vel, odor)
            viewer.update()
        
        # Progress
        if step % 5000 == 0 and step > 0:
            logger.info(f"Step {step}/{n_steps} - Pos: {pos}, Vel: {np.linalg.norm(vel):.3f} mm/s, Odor: {odor:.3f}")
    
    logger.info("Simulation complete. Close window to exit.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=10, help='Duration in seconds')
    args = parser.parse_args()
    
    run_interactive_simulation(args.duration)
