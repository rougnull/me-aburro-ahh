"""
Sensory transduction: converts physical odor measurements to neural inputs.
Simulates olfactory receptor binding and signal transduction.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SensoryTransduction:
    """
    Converts environmental stimuli (odor concentration, touch, etc.)
    into neural activity (receptor currents for computing models).
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize sensory transduction model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Receptor binding kinetics (simplified Hill equation)
        self.kd = 0.5  # Dissociation constant (odor concentration)
        self.n = 1.5   # Hill coefficient (cooperativity)
        
        # Adaptation (fatigue) of receptors
        self.adaptation_tau = 5.0  # Time constant of adaptation (s)
        self.adaptation_state = np.zeros(50)  # One per ORN type
        
        logger.info("SensoryTransduction initialized")
    
    def odor_to_current(self, odor_concentration: float, 
                       adaptation_level: float = 0.0) -> np.ndarray:
        """
        Convert odor concentration to receptor current using Hill equation.
        
        Args:
            odor_concentration: Odor concentration (0.0 to 1.0)
            adaptation_level: Level of receptor adaptation (0.0 to 1.0)
            
        Returns:
            Array of receptor currents (one per ORN type)
        """
        
        # Hill equation for receptor binding
        # response = [odor]^n / (kd^n + [odor]^n)
        numerator = odor_concentration ** self.n
        denominator = self.kd ** self.n + numerator
        
        response = numerator / denominator
        
        # Adaptation reduces the response
        adapted_response = response * (1.0 - adaptation_level * 0.8)
        
        # Heterogeneous ORN responses (some neurons prefer different odors)
        n_orn_types = 50
        orn_tuning = np.random.exponential(0.5, n_orn_types)
        
        # Current is proportional to binding and ORN type
        current = adapted_response * orn_tuning * 100  # Scaling factor
        
        return current
    
    def update_adaptation(self, odor_concentration: float, dt: float):
        """
        Update receptor adaptation (slowly)
        
        Args:
            odor_concentration: Current odor level
            dt: Timestep
        """
        # Adaptation increases with odor exposure
        target_adaptation = odor_concentration
        
        for i in range(len(self.adaptation_state)):
            self.adaptation_state[i] += (target_adaptation - self.adaptation_state[i]) * dt / self.adaptation_tau
    
    def mechanoreceptor_input(self, leg_forces: np.ndarray) -> np.ndarray:
        """
        Convert mechanical forces to mechanoreceptor currents.
        (For future implementation of tactile feedback)
        
        Args:
            leg_forces: Force vector for each leg segment
            
        Returns:
            Mechanoreceptor currents
        """
        # Placeholder for future implementation
        return np.abs(leg_forces) * 50  # Simple force-to-current scaling


class DescendingInterface:
    """
    Converts descending neuron activity (motor commands from the brain)
    into motor control signals for the legs.
    
    DN firing patterns encode:
    - Walking direction and speed
    - Turning rate
    - Wing modulation
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the descending interface."""
        self.config = config or {}
        
        # Motor command thresholds and gains
        self.movement_threshold = 0.1
        self.forward_gain = self.config.get('motor_gains', {}).get('forward_speed', 20.0)
        self.rotation_gain = self.config.get('motor_gains', {}).get('rotation_speed', 45.0)
        
        logger.info("DescendingInterface initialized")
    
    def dn_activity_to_motor_commands(self, dn_activity: np.ndarray,
                                     dn_spikes: np.ndarray) -> Dict:
        """
        Decode descending neuron activity into motor commands.
        
        Assumption: DNs are organized into functional groups:
        - DN0-DN2: Forward locomotion
        - DN3-DN5: Left turn
        - DN6-DN9: Right turn
        
        Args:
            dn_activity: Membrane potentials of DNs
            dn_spikes: Binary spike activity
            
        Returns:
            Dictionary with motor commands
        """
        
        if len(dn_activity) < 10:
            raise ValueError("DN layer too small for standard decoding")
        
        # Average activity of each functional group
        forward_activity = np.mean(dn_activity[:3])
        left_activity = np.mean(dn_activity[3:6])
        right_activity = np.mean(dn_activity[6:10])
        
        # Normalize to [-1, 1] range (assuming resting potential ~-70 mV, threshold ~-50 mV)
        forward_normalized = np.clip((forward_activity + 60) / 20, -1, 1)
        left_normalized = np.clip((left_activity + 60) / 20, -1, 1)
        right_normalized = np.clip((right_activity + 60) / 20, -1, 1)
        
        # Compute motor commands
        forward_speed = max(0, forward_normalized) * self.forward_gain
        
        # Turning (left is positive)
        turn_speed = (right_normalized - left_normalized) * self.rotation_gain
        
        return {
            'forward_speed': forward_speed,
            'angular_velocity': turn_speed,
            'spike_rate': np.mean(dn_spikes.astype(float)) * 100  # Hz
        }
    
    def modulate_cpg_command(self, forward_speed: float, turn_speed: float) -> Dict:
        """
        Modulate central pattern generator (CPG) parameters based on DN commands.
        
        The CPG handles the actual leg rhythms; DN commands just modulate it.
        
        Args:
            forward_speed: Forward velocity command (mm/s)
            turn_speed: Turning velocity command (deg/s)
            
        Returns:
            CPG parameters (frequency, amplitude)
        """
        
        # Base CPG frequency (~10 Hz for walking)
        base_frequency = 10.0
        
        # CPG frequency increases with forward speed
        cpg_frequency = base_frequency * (1.0 + forward_speed / 20.0)
        
        # Asymmetry modulates turning
        cpg_amplitude = 1.0
        if turn_speed > 0:  # Left turn
            left_amplitude = 1.0
            right_amplitude = max(0.3, 1.0 - abs(turn_speed) / 45.0)
        elif turn_speed < 0:  # Right turn
            left_amplitude = max(0.3, 1.0 + turn_speed / 45.0)
            right_amplitude = 1.0
        else:  # Straight
            left_amplitude = 1.0
            right_amplitude = 1.0
        
        return {
            'frequency': cpg_frequency,
            'left_amplitude': left_amplitude,
            'right_amplitude': right_amplitude
        }
