"""
Body interface module - wrapper for NeuroMechFly robot.
Handles low-level motor commands and physics integration.
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FlyInterface:
    """
    Wrapper interface for NeuroMechFly robot.
    Abstracts the low-level API for motor control and state readout.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the fly robot interface.
        
        Args:
            config: Configuration dictionary with fly parameters
        """
        self.config = config or {}
        
        # TODO: Initialize actual NeuroMechFly simulation here
        # from neuromechfly import Fly
        # self.fly = Fly(...)
        
        # For now, use a mock implementation
        self.position = np.array([50.0, 50.0, 0.0])  # Start in arena center
        self.orientation = np.array([1.0, 0.0, 0.0])  # Facing forward
        self.velocity = np.array([0.0, 0.0, 0.0])
        
        self.motor_commands = {'forward_speed': 0.0, 'angular_velocity': 0.0}
        self.walking_controller = WalkingController(config)
        
        logger.info("FlyInterface initialized (mock mode)")
    
    def get_body_position(self) -> np.ndarray:
        """Get current 3D position of fly center of mass."""
        return self.position.copy()
    
    def get_head_position(self) -> np.ndarray:
        """Get 3D position of fly head (offset from COM)."""
        # Simple offset in direction of orientation
        head_offset = self.orientation * 2.5  # 2.5mm ahead
        return self.position + head_offset
    
    def get_orientation(self) -> np.ndarray:
        """Get current heading direction (unit vector)."""
        return self.orientation.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity vector."""
        return self.velocity.copy()
    
    def apply_motor_commands(self, commands: Dict):
        """
        Apply motor commands to the fly.
        
        Args:
            commands: Dict with 'forward_speed' and 'angular_velocity'
        """
        self.motor_commands = commands
        self.walking_controller.set_command(commands)
    
    def physics_step(self, dt: float):
        """
        Execute one physics simulation step.
        
        Args:
            dt: Timestep (seconds)
        """
        # Get joint torques from walking controller
        torques = self.walking_controller.get_action(dt)
        
        # TODO: Apply torques to NeuroMechFly simulation
        # self.fly.apply_torques(torques)
        # self.fly.step(dt)
        
        # For now, update position based on commands
        forward_speed = self.motor_commands.get('forward_speed', 0.0)
        angular_velocity = self.motor_commands.get('angular_velocity', 0.0)
        
        # Update orientation (rotation around z-axis)
        angle_change = angular_velocity * dt * np.pi / 180  # Convert deg/s to rad/s
        rotation_matrix = np.array([
            [np.cos(angle_change), -np.sin(angle_change), 0],
            [np.sin(angle_change), np.cos(angle_change), 0],
            [0, 0, 1]
        ])
        
        self.orientation = rotation_matrix @ self.orientation
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
        
        # Update position
        self.velocity = self.orientation * forward_speed / 1000  # Convert mm/s to m/s
        self.position += self.velocity * dt
        
        # Keep in bounds (simple wraparound)
        arena_size = 100  # mm
        self.position[:2] = np.mod(self.position[:2], arena_size)


class WalkingController:
    """
    Central Pattern Generator (CPG) based walking controller.
    Generates rhythmic leg movements modulated by descending commands.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the walking controller.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # CPG parameters
        self.cpg_frequency = 10.0  # Hz
        self.cpg_phase = 0.0
        self.leg_amplitudes = np.ones(6) * 1.0  # 6 legs
        
        # Command state
        self.forward_command = 0.0
        self.turn_command = 0.0
        
        logger.info("WalkingController initialized")
    
    def set_command(self, commands: Dict):
        """
        Set motor command inputs.
        
        Args:
            commands: Dict with motor commands from brain
        """
        self.forward_command = commands.get('forward_speed', 0.0)
        self.turn_command = commands.get('angular_velocity', 0.0)
    
    def get_action(self, dt: float) -> np.ndarray:
        """
        Generate joint torques for all leg joints.
        
        Args:
            dt: Timestep
            
        Returns:
            Array of joint torques
        """
        # Update CPG phase
        self.cpg_phase += 2 * np.pi * self.cpg_frequency * dt
        self.cpg_phase = self.cpg_phase % (2 * np.pi)
        
        # Modulate CPG parameters based on commands
        if abs(self.forward_command) > 0.1:
            self.cpg_frequency = 10.0 + abs(self.forward_command) / 5
        else:
            self.cpg_frequency = 10.0
        
        # Asymmetric CPG for turning
        left_amplitude = 1.0
        right_amplitude = 1.0
        
        if abs(self.turn_command) > 1.0:
            if self.turn_command > 0:  # Left turn
                left_amplitude = 1.0
                right_amplitude = max(0.3, 1.0 - abs(self.turn_command) / 45)
            else:  # Right turn
                left_amplitude = max(0.3, 1.0 + self.turn_command / 45)
                right_amplitude = 1.0
        
        # Generate sinusoidal leg movements
        num_joints = 18  # 3 pairs * 3 joints per leg
        torques = np.zeros(num_joints)
        
        for leg_idx in range(6):
            for joint in range(3):
                idx = leg_idx * 3 + joint
                
                # Left legs (indices 0-8)
                if leg_idx < 3:
                    amplitude = left_amplitude
                    phase_offset = leg_idx * 2 * np.pi / 3  # Phase between legs
                else:
                    amplitude = right_amplitude
                    phase_offset = (leg_idx - 3) * 2 * np.pi / 3
                
                # Simple sinusoidal output
                torques[idx] = amplitude * np.sin(self.cpg_phase + phase_offset)
        
        return torques
