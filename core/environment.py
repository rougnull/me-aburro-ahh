"""
Environment module for arena simulation with odor gradients.
Simulates the spatial environment where the fly moves.
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class Arena:
    """
    Simulates a 3D arena environment with odor sources.
    """
    
    def __init__(self, width: float, height: float, depth: float,
                 food_position: Tuple[float], food_intensity: float = 1.0,
                 diffusion_coeff: float = 0.1, decay_rate: float = 0.05):
        """
        Initialize the arena.
        
        Args:
            width, height, depth: Arena dimensions (mm)
            food_position: Position of food source (x, y, z)
            food_intensity: Maximum odor concentration
            diffusion_coeff: Odor diffusion coefficient
            decay_rate: Odor decay rate
        """
        self.width = width
        self.height = height
        self.depth = depth
        
        self.food_position = np.array(food_position)
        self.food_intensity = food_intensity
        self.diffusion_coeff = diffusion_coeff
        self.decay_rate = decay_rate
        
        # Wind parameters
        self.wind_speed = 0.0
        self.wind_direction = np.array([1.0, 0.0, 0.0])
        
        logger.info(f"Arena initialized: {width}x{height}x{depth} mm")
    
    def get_odor_concentration(self, position: np.ndarray) -> float:
        """
        Calculate odor concentration at a given position.
        Uses Gaussian distribution centered at food source.
        
        Args:
            position: (x, y, z) position in the arena
            
        Returns:
            Odor concentration (0.0 to 1.0)
        """
        # Clip position to arena bounds
        pos = np.clip(position[:2], [0, 0], [self.width, self.height])
        
        # Distance to food
        distance = np.linalg.norm(pos - self.food_position[:2])
        
        # Gaussian distribution (peak at food source)
        # Falloff distance is ~25% of arena width
        sigma = self.width / 4.0
        
        concentration = self.food_intensity * np.exp(-(distance**2) / (2 * sigma**2))
        
        # Add wind-aided plume (if implemented)
        concentration *= (1.0 + self.wind_speed * 0.1)
        
        return float(np.clip(concentration, 0.0, 1.0))
    
    def get_odor_gradient(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate the odor gradient (direction of steepest ascent).
        
        Args:
            position: Current position
            
        Returns:
            Gradient vector (olfactory cue for navigation)
        """
        delta = 0.5  # Small delta for numerical gradient
        
        odor_center = self.get_odor_concentration(position)
        
        # X component
        pos_x = position.copy()
        pos_x[0] += delta
        odor_x = self.get_odor_concentration(pos_x)
        
        # Y component
        pos_y = position.copy()
        pos_y[1] += delta
        odor_y = self.get_odor_concentration(pos_y)
        
        gradient = np.array([
            (odor_x - odor_center) / delta,
            (odor_y - odor_center) / delta
        ])
        
        return gradient
    
    def set_wind(self, speed: float, direction: np.ndarray):
        """
        Set wind parameters in the arena.
        
        Args:
            speed: Wind speed (m/s)
            direction: Unit vector for wind direction
        """
        self.wind_speed = speed
        self.wind_direction = direction / np.linalg.norm(direction)
    
    def is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within arena bounds."""
        return (0 <= position[0] <= self.width and
                0 <= position[1] <= self.height and
                0 <= position[2] <= self.depth)
