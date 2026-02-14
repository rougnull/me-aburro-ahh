#!/usr/bin/env python3
"""
Main entry point for NeuroMechFly simulation.

Usage:
    python run_experiment.py --config config/default.yaml --duration 60
    
This script orchestrates:
1. Loading configuration
2. Initializing the fly robot, brain, and environment
3. Running the main simulation loop
4. Saving results and generating visualizations
"""

import argparse
import logging
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Import project modules
from core.simulation import NeuroMechFlySimulation
from core.environment import Arena
from brain.olfactory_circuit import OlfactoryCircuit
from body.fly_interface import FlyInterface
from analysis.visualization import plot_trajectory, plot_neural_activity, plot_odor_response
from analysis.visualization_3d import ArenaVisualizer3D


def setup_logging(log_level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    return config


def merge_configs(base_config: dict, environment_config: dict, 
                 fly_config: dict, brain_config: dict) -> dict:
    """Merge multiple config files into one."""
    config = {'base': base_config}
    config.update(environment_config)
    config.update(brain_config)
    config['fly_params'] = fly_config
    return config


def main(args):
    """Main simulation function."""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("NeuroMechFly Embodied Simulation")
    logger.info("=" * 70)
    
    # Load configurations
    logger.info(f"Loading configuration from {args.config}")
    
    config_dir = Path(args.config).parent
    
    env_config = load_config(config_dir / 'environment.yaml')
    fly_config = load_config(config_dir / 'fly_params.yaml')
    brain_config = load_config(config_dir / 'brain_params.yaml')
    
    config = merge_configs({}, env_config, fly_config, brain_config)
    
    logger.info(f"Configuration loaded: {len(config)} sections")
    
    # Initialize components
    logger.info("Initializing simulation components...")
    
    # Environment (arena with odor)
    env_params = config.get('arena', {})
    odor_params = config.get('odor', {})
    
    arena = Arena(
        width=env_params.get('width', 100.0),
        height=env_params.get('height', 100.0),
        depth=env_params.get('depth', 50.0),
        food_position=odor_params.get('food_position', [50, 50, 0]),
        food_intensity=odor_params.get('food_intensity', 1.0),
        diffusion_coeff=odor_params.get('diffusion_coefficient', 0.1),
        decay_rate=odor_params.get('decay_rate', 0.05)
    )
    
    # Neural model
    brain = OlfactoryCircuit(config)
    
    # Robot body
    fly = FlyInterface(config)
    
    # Main simulation
    sim = NeuroMechFlySimulation(fly, brain, arena, config)
    
    logger.info("All components initialized successfully")
    
    # Run simulation
    num_steps = int(args.duration * 1000)  # Convert seconds to ms timesteps (0.001s each)
    logger.info(f"Running simulation for {args.duration}s ({num_steps} steps)...")
    
    try:
        sim.run(num_steps, verbose=True)
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        return 1
    
    # Save results
    logger.info("Saving results...")
    
    output_dir = Path('data') / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data as HDF5
    data_file = output_dir / 'simulation_data.h5'
    sim.save_data(str(data_file))
    
    # Generate plots
    logger.info("Generating visualizations...")
    logged_data = sim.get_logged_data()
    
    try:
        plot_trajectory(logged_data, str(output_dir / 'trajectory.png'))
        plot_neural_activity(logged_data, str(output_dir / 'neural_activity.png'))
        plot_odor_response(logged_data, str(output_dir / 'odor_response.png'))
        
        # Generate 3D visualizations
        viz_3d = ArenaVisualizer3D(config)
        viz_3d.plot_3d_trajectory(logged_data, str(output_dir / '3d_trajectory.png'), with_odor_field=True)
        viz_3d.plot_neural_heatmap(logged_data, str(output_dir / 'neural_heatmap.png'))
        viz_3d.plot_behavior_analysis(logged_data, str(output_dir / 'behavior_analysis.png'))
        
    except Exception as e:
        logger.warning(f"Error generating plots: {e}")
    
    # Summary statistics
    logger.info("=" * 70)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 70)
    
    positions = np.array(logged_data.get('position', []))
    if positions.size > 0:
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        logger.info(f"Total distance traveled: {total_distance:.1f} mm")
        logger.info(f"Final position: ({positions[-1, 0]:.1f}, {positions[-1, 1]:.1f})")
        
        # Distance to food
        food_pos_full = np.array(odor_params.get('food_position', [50, 50, 0]))
        food_pos = food_pos_full[:2]  # Take only x, y
        final_dist_to_food = np.linalg.norm(positions[-1, :2] - food_pos)
        logger.info(f"Final distance to food: {final_dist_to_food:.1f} mm")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info("")
    logger.info("Visualización disponible:")
    logger.info(f"  - Trayectoria: {output_dir / 'trajectory.png'}")
    logger.info(f"  - Actividad neural: {output_dir / 'neural_activity.png'}")
    logger.info(f"  - Respuesta olfativa: {output_dir / 'odor_response.png'}")
    logger.info(f"  - Datos crudos: {output_dir / 'simulation_data.h5'}")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuroMechFly embodied simulation')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/environment.yaml',
        help='Path to main config file'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Simulation duration in seconds'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    exit(exit_code)
