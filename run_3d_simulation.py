#!/usr/bin/env python3
"""
Integrated NeuroMechFly 3D Embodied Simulation with Real-Time Visualization.

This script combines:
- Realistic fly body (skeleton kinematics)
- Neural circuit (olfactory + decision-making)
- Interactive 3D visualization (Vispy)

Usage:
    python run_3d_simulation.py --duration 10 --display
"""

import sys
import os
import argparse
import numpy as np
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.simulation import NeuroMechFlySimulation
from core.environment import Arena
from brain.olfactory_circuit import OlfactoryCircuit
from body.realistic_body import FlyBody, RealisticFlyInterface
from analysis.vispy_viewer import FlyVisualizer3D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbodiedSimulation:
    """Integrated embodied simulation with neural control and realistic body."""
    
    def __init__(self, config_path: str = None, display: bool = True):
        """
        Initialize embodied simulation.
        
        Args:
            config_path: Path to config YAML
            display: Whether to show real-time visualization
        """
        # Load config
        if config_path is None:
            config_path = project_root / 'config' / 'environment.yaml'
        
        config_dir = Path(config_path).parent
        
        # Load all config files
        with open(config_dir / 'environment.yaml', 'r') as f:
            env_config = yaml.safe_load(f) or {}
        with open(config_dir / 'fly_params.yaml', 'r') as f:
            fly_config = yaml.safe_load(f) or {}
        with open(config_dir / 'brain_params.yaml', 'r') as f:
            brain_config = yaml.safe_load(f) or {}
        
        # Merge configs
        self.config = {'base': {}}
        self.config.update(env_config)
        self.config.update(brain_config)
        self.config['fly_params'] = fly_config
        
        # Initialize components
        logger.info("Initializing arena environment...")
        env_params = self.config.get('arena', {})
        odor_params = self.config.get('odor', {})
        
        arena = Arena(
            width=env_params.get('width', 100.0),
            height=env_params.get('height', 100.0),
            depth=env_params.get('depth', 50.0),
            food_position=odor_params.get('food_position', [50, 50, 0]),
            food_intensity=odor_params.get('food_intensity', 1.0),
            diffusion_coeff=odor_params.get('diffusion_coefficient', 0.1),
            decay_rate=odor_params.get('decay_rate', 0.05)
        )
        
        logger.info("Initializing neural circuit...")
        brain = OlfactoryCircuit(self.config)
        
        logger.info("Initializing realistic fly body...")
        self.fly_body = FlyBody()
        fly_interface = RealisticFlyInterface(self.fly_body)
        
        # Create simulation
        logger.info("Initializing simulator...")
        self.simulation = NeuroMechFlySimulation(fly_interface, brain, arena, self.config)
        
        # Setup visualization
        self.display = display
        self.visualizer = None
        if display:
            logger.info("Initializing 3D visualizer...")
            self.visualizer = FlyVisualizer3D(self.config)
        
        # State tracking
        self.step_count = 0
        self.visualization_interval = 5  # Update viz every N steps
        
        logger.info("Embodied simulation ready!")
    
    def step(self) -> Dict:
        """
        Execute one simulation step.
        
        Returns:
            State dictionary with all components
        """
        # Execute neural + motor step
        self.simulation.step()
        self.step_count += 1
        
        # Get current state
        pos = self.simulation.fly.get_body_position()
        odor_conc = self.simulation.env.get_odor_concentration(pos)
        
        state = {
            'position': pos.copy(),
            'heading': self.simulation.fly.get_orientation().copy(),
            'odor': np.array([odor_conc]),
            'motor_command': self.simulation.fly.get_velocity().copy(),
        }
        
        # Update visualization
        if self.display and self.step_count % self.visualization_interval == 0:
            body_frame = self.fly_body.get_body_frame()
            self.visualizer.render_fly(body_frame)
            
            # Add trajectory point
            self.visualizer.add_trajectory_point(pos, odor_conc)
        
        return state
    
    def run(self, duration: float):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation time in seconds
        """
        dt = self.simulation.timestep
        n_steps = int(duration / dt)
        logger.info(f"Running {n_steps} steps ({duration}s at {dt}s timestep)...")
        
        # Collect statistics
        positions = []
        odors = []
        velocities = []
        
        try:
            for step in range(n_steps):
                state = self.step()
                
                positions.append(state['position'])
                odors.append(state['odor'][0] if len(state['odor']) > 0 else 0)
                velocities.append(np.linalg.norm(state['motor_command'][:2]))
                
                # Progress
                if step % 5000 == 0 and step > 0:
                    avg_vel = np.mean(velocities[-100:])
                    logger.info(f"Step {step}/{n_steps} - Position: {state['position']}, "
                              f"Avg velocity: {avg_vel:.3f} mm/s")
                
                # Handle visualization events
                if self.display:
                    self.visualizer.update()
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        
        finally:
            # Finalize trajectory display
            if self.display:
                self.visualizer.update_trajectory_display()
                logger.info("3D visualization complete. Close window to finish.")
                # Keep window open
                try:
                    from vispy.app import run as vispy_run
                    vispy_run()
                except:
                    pass
        
        # Return statistics
        return {
            'positions': np.array(positions),
            'odors': np.array(odors),
            'velocities': np.array(velocities),
            'n_steps': n_steps,
            'duration': duration,
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run integrated 3D NeuroMechFly embodied simulation"
    )
    parser.add_argument('--duration', type=float, default=10,
                       help='Simulation duration in seconds (default: 10)')
    parser.add_argument('--display', action='store_true', default=True,
                       help='Show real-time 3D visualization')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Run simulation
    display = args.display and not args.no_display
    
    sim = EmbodiedSimulation(config_path=args.config, display=display)
    results = sim.run(duration=args.duration)
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Duration: {results['duration']:.1f}s ({results['n_steps']} steps)")
    print(f"Mean velocity: {np.mean(results['velocities']):.3f} mm/s")
    print(f"Max velocity: {np.max(results['velocities']):.3f} mm/s")
    print(f"Distance traveled: {np.sum(results['velocities']) * 0.001:.2f} mm")
    print(f"Mean odor detected: {np.mean(results['odors']):.3f}")
    print("="*60)


if __name__ == '__main__':
    main()
