"""
Enhanced Fly Body Model with realistic skeleton.
Implements a 3D fly body with realistic kinematics and forward dynamics.
Compatible with future NeuroMechFly integration.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FlyBody:
    """
    3D fly body model with realistic skeletal structure.
    
    Skeletal structure:
    - Head (vision, sensors)
    - Thorax (central motor hub)
    - Abdomen (balance, center of mass)
    - 6 legs with 3 segments each
    - 2 wings
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize fly body model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Body segments (position in 3D)
        self.head_pos = np.array([0.0, 0.0, 0.0])
        self.thorax_pos = np.array([0.0, 0.0, 0.0])
        self.abdomen_pos = np.array([0.0, 0.0, -2.0])  # Slightly behind
        
        # Center of mass (total body frame)
        self.com_pos = np.array([0.0, 0.0, 0.0])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        
        # Orientation (quaternion or Euler angles)
        self.heading = np.array([1.0, 0.0, 0.0])  # Forward direction
        self.pitch = 0.0  # Pitch angle
        self.roll = 0.0   # Roll angle
        
        # Angular velocity
        self.ang_vel = np.array([0.0, 0.0, 0.0])  # [pitch_rate, roll_rate, yaw_rate]
        
        # Leg states (6 legs: Front-L/R, Middle-L/R, Hind-L/R)
        self.num_legs = 6
        self.leg_segments = 3  # Coxa, femur, tibia
        self.leg_positions = self._initialize_legs()  # (6, 3, 3) - 6 legs, 3 segments, 3D pos
        self.leg_angles = np.zeros((6, 3))  # Joint angles (degrees)
        
        # Wing states
        self.wing_angle_left = 0.0   # Degrees
        self.wing_angle_right = 0.0
        self.wing_beat_freq = 200.0  # Hz
        self.wing_beat_phase = 0.0   # Current phase
        
        # Physical parameters
        self.body_mass = 0.001  # kg (~1 mg)
        self.body_length = 5.0  # mm
        self.body_width = 2.0   # mm
        
        # Control parameters
        self.forward_cmd = 0.0   # Forward velocity command (mm/s)
        self.turn_cmd = 0.0      # Turn rate command (deg/s)
        
        # Joint limits
        self.leg_joint_limits = [
            (-30, 30),   # Coxa (horizontal swing)
            (-90, 45),   # Femur (up-down)
            (-45, 90),   # Tibia (extension-flexion)
        ]
        
        # Leg contact with ground
        self.leg_contact = np.zeros(6, dtype=bool)  # Which legs touch ground
        
        logger.info("FlyBody initialized with realistic skeleton")
    
    def reset(self):
        """Reset body to initial state."""
        self.com_pos = np.array([0.0, 0.0, 0.0])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.heading = np.array([1.0, 0.0, 0.0])
        self.pitch = 0.0
        self.roll = 0.0
        self.ang_vel = np.array([0.0, 0.0, 0.0])
        self.leg_angles = np.zeros((6, 3))
        self.wing_angle_left = 0.0
        self.wing_angle_right = 0.0
        self.forward_cmd = 0.0
        self.turn_cmd = 0.0
        logger.info("FlyBody reset to initial state")
    
    def _initialize_legs(self) -> np.ndarray:
        """Initialize leg positions in 3D space."""
        # Leg attachment points on thorax (in body frame)
        leg_bases = np.array([
            [1.0, 1.5, 0.0],   # Front-Left
            [-1.0, 1.5, 0.0],  # Front-Right
            [2.0, 0.0, 0.0],   # Middle-Left
            [-2.0, 0.0, 0.0],  # Middle-Right
            [1.5, -1.5, 0.0],  # Hind-Left
            [-1.5, -1.5, 0.0], # Hind-Right
        ])
        
        # Initialize leg positions (extended downward)
        legs = np.zeros((6, 3, 3))
        
        for leg_id in range(6):
            # Segment lengths (mm)
            coxa_len = 1.0
            femur_len = 2.0
            tibia_len = 2.5
            
            # Rest position: legs extended downward
            legs[leg_id, 0] = leg_bases[leg_id]  # Coxa (at thorax)
            legs[leg_id, 1] = leg_bases[leg_id] + np.array([0, 0, -femur_len])  # Femur
            legs[leg_id, 2] = legs[leg_id, 1] + np.array([0, 0, -tibia_len])  # Tibia
        
        return legs
    
    def step(self, dt: float, motor_commands: Dict):
        """
        Update fly body state for one timestep.
        
        Args:
            dt: Timestep (seconds)
            motor_commands: Dict with 'forward_speed' and 'angular_velocity'
        """
        # Parse motor commands
        self.forward_cmd = motor_commands.get('forward_speed', 0.0)
        self.turn_cmd = motor_commands.get('angular_velocity', 0.0)
        
        # Update orientation based on turn command
        yaw_rate = self.turn_cmd * np.pi / 180  # Convert to rad/s
        self._update_orientation(dt, yaw_rate)
        
        # Update velocity based on forward command
        forward_vel = self.forward_cmd / 1000  # Convert mm/s to m/s
        self.com_vel = self.heading * forward_vel
        
        # Update position
        self.com_pos += self.com_vel * dt
        
        # Update leg kinematics (CPG-driven)
        self._update_leg_kinematics(dt)
        
        # Update wing kinematics
        self._update_wing_kinematics(dt)
        
        # Apply ground contact forces (if legs touch ground, damping)
        self._apply_ground_contact(dt)
    
    def _update_orientation(self, dt: float, yaw_rate: float):
        """Update fly heading based on yaw rate."""
        # Rotate heading vector around z-axis
        angle = yaw_rate * dt
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 2D rotation in XY plane
        x = self.heading[0]
        y = self.heading[1]
        
        self.heading[0] = x * cos_a - y * sin_a
        self.heading[1] = x * sin_a + y * cos_a
        self.heading = self.heading / np.linalg.norm(self.heading)  # Normalize
    
    def _update_leg_kinematics(self, dt: float):
        """Update leg joint angles based on CPG pattern."""
        # Simple CPG: legs move in tripod pattern (Front+Hind alternating with Middle)
        cpg_freq = 10.0  # Hz (typical walking frequency)
        cpg_phase = (np.sin(2 * np.pi * cpg_freq * dt) + 1) / 2  # 0 to 1
        
        for leg_id in range(6):
            # Tripod pattern: legs {0,4,3} vs {1,5,2}
            is_frontmiddle = leg_id in [0, 4, 3]  # Actually front-hind pairs
            
            # Power stroke vs return stroke
            if (cpg_phase < 0.5 and is_frontmiddle) or (cpg_phase >= 0.5 and not is_frontmiddle):
                # Power stroke: leg extends forward
                swing_amount = cpg_phase * 30
            else:
                # Return stroke: leg retracts
                swing_amount = (1 - cpg_phase) * 30
            
            # Update angles
            self.leg_angles[leg_id, 0] = np.clip(swing_amount - 15, -30, 30)  # Coxa swing
            self.leg_angles[leg_id, 1] = -60 + swing_amount  # Femur (up-down)
            self.leg_angles[leg_id, 2] = 30 + swing_amount * 0.5  # Tibia
        
        # Update leg positions from angles
        self._forward_kinematics_legs()
    
    def _forward_kinematics_legs(self):
        """Calculate 3D leg end-effector positions from joint angles."""
        for leg_id in range(6):
            angles = self.leg_angles[leg_id]
            
            # Segment lengths
            coxa_len = 1.0
            femur_len = 2.0
            tibia_len = 2.5
            
            # Get attachment point
            leg_base = self.leg_positions[leg_id, 0]
            
            # Forward kinematics: convert angles to 3D positions
            # Simplified model: angles in degrees
            theta1, theta2, theta3 = np.radians(angles[0]), np.radians(angles[1]), np.radians(angles[2])
            
            # Coxa segment
            p1 = leg_base + np.array([
                coxa_len * np.cos(theta1),
                coxa_len * np.sin(theta1),
                0
            ])
            
            # Femur segment
            p2 = p1 + np.array([
                femur_len * np.cos(theta2),
                0,
                -femur_len * np.sin(theta2)
            ])
            
            # Tibia segment
            p3 = p2 + np.array([
                tibia_len * np.cos(theta3),
                0,
                -tibia_len * np.sin(theta3)
            ])
            
            # Update leg positions
            self.leg_positions[leg_id, 0] = p1
            self.leg_positions[leg_id, 1] = p2
            self.leg_positions[leg_id, 2] = p3  # End effector (claw)
            
            # Check ground contact (if z < 0)
            self.leg_contact[leg_id] = p3[2] < -5.0
    
    def _update_wing_kinematics(self, dt: float):
        """Update wing beat pattern."""
        # Wing beat is independent of walking
        self.wing_beat_phase += 2 * np.pi * self.wing_beat_freq * dt
        self.wing_beat_phase = self.wing_beat_phase % (2 * np.pi)
        
        # Wing beat amplitude
        amplitude = 30.0  # degrees
        self.wing_angle_left = amplitude * np.sin(self.wing_beat_phase)
        self.wing_angle_right = amplitude * np.sin(self.wing_beat_phase + np.pi)  # Opposite phase
    
    def _apply_ground_contact(self, dt: float):
        """Apply damping if legs are in contact with ground."""
        if np.any(self.leg_contact):
            # Reduce velocity (friction)
            self.com_vel *= 0.98
    
    def get_body_frame(self) -> Dict:
        """Get current body state as dict for visualization."""
        return {
            'com_pos': self.com_pos.copy(),
            'heading': self.heading.copy(),
            'com_vel': self.com_vel.copy(),
            'legs': self.leg_positions.copy(),
            'wing_left': self.wing_angle_left,
            'wing_right': self.wing_angle_right,
            'head_pos': self.com_pos + self.heading * 2.5,  # Head ahead of COM
            'thorax_pos': self.com_pos,
            'abdomen_pos': self.com_pos - self.heading * 1.5,
        }


class RealisticFlyInterface:
    """
    Enhanced fly interface using realistic body model.
    Compatible with NeuroMechFly when available.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize realistic fly interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.body = FlyBody(config)
        self.walking_controller = EnhancedWalkingController(config)
        
        # State tracking
        self.position = self.body.com_pos.copy()
        self.orientation = self.body.heading.copy()
        self.velocity = self.body.com_vel.copy()
        
        logger.info("RealisticFlyInterface initialized with skeleton-based model")
    
    def reset_to_initial_state(self):
        """Reset fly to initial position and state."""
        self.body.reset()
        self.position = self.body.com_pos.copy()
        self.orientation = self.body.heading.copy()
        self.velocity = self.body.com_vel.copy()
        logger.info("RealisticFlyInterface reset to initial state")
    
    def get_body_position(self) -> np.ndarray:
        """Get current 3D position of fly COM."""
        return self.body.com_pos.copy()
    
    def get_head_position(self) -> np.ndarray:
        """Get 3D position of fly head."""
        frame = self.body.get_body_frame()
        return frame['head_pos']
    
    def get_orientation(self) -> np.ndarray:
        """Get current heading direction."""
        return self.body.heading.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity vector."""
        return self.body.com_vel.copy()
    
    def apply_motor_commands(self, commands: Dict):
        """Apply motor commands to fly."""
        self.walking_controller.set_command(commands)
    
    def physics_step(self, dt: float):
        """
        Execute one physics simulation step.
        
        Args:
            dt: Timestep (seconds)
        """
        # Get joint commands from walking controller
        motor_command = self.walking_controller.get_action(dt)
        
        # Update body kinematics
        self.body.step(dt, motor_command)
        
        # Update cached state
        self.position = self.body.com_pos.copy()
        self.orientation = self.body.heading.copy()
        self.velocity = self.body.com_vel.copy()
        
        # Keep in arena bounds (wraparound)
        arena_size = 100  # mm
        self.position[0] = self.position[0] % arena_size
        self.position[1] = self.position[1] % arena_size
    
    def get_body_frame(self) -> Dict:
        """Get complete body frame for rendering."""
        return self.body.get_body_frame()


class EnhancedWalkingController:
    """
    Enhanced CPG (Central Pattern Generator) with better dynamics.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize walking controller."""
        self.config = config or {}
        self.cpg_frequency = 10.0
        self.forward_command = 0.0
        self.turn_command = 0.0
        
    def set_command(self, commands: Dict):
        """Set motor commands."""
        self.forward_command = commands.get('forward_speed', 0.0)
        self.turn_command = commands.get('angular_velocity', 0.0)
    
    def get_action(self, dt: float) -> Dict:
        """Generate motor commands."""
        return {
            'forward_speed': self.forward_command,
            'angular_velocity': self.turn_command
        }
