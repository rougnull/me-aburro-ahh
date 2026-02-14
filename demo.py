#!/usr/bin/env python3
"""
Quick demo script for NeuroMechFly simulation.
Shows basic usage without requiring full configuration.

Usage:
    python demo.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import logging
from core.simulation import NeuroMechFlySimulation
from core.environment import Arena
from brain.olfactory_circuit import OlfactoryCircuit
from body.fly_interface import FlyInterface

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_config():
    """Create a minimal configuration for demo."""
    config = {
        # Environment
        'arena': {
            'width': 100.0,
            'height': 100.0,
            'depth': 50.0
        },
        'odor': {
            'food_position': [50.0, 50.0, 0.0],
            'food_intensity': 1.0,
            'diffusion_coefficient': 0.1,
            'decay_rate': 0.05
        },
        'wind': {
            'speed': 0.0,
            'direction': [1.0, 0.0, 0.0]
        },
        'physics': {
            'timestep': 0.001,
            'gravity': [0.0, 0.0, -9.81]
        },
        # Flying insect parameters
        'neurons': {
            'orn_count': 50,
            'pn_count': 20,
            'kc_count': 2000,
            'mbon_count': 34,
            'dn_count': 10
        },
        'synapses': {
            'orn_to_pn_weight': 0.5,
            'pn_to_kc_weight': 0.1,
            'kc_to_mbon_weight': 0.2,
            'mbon_to_dn_weight': 0.3,
            'lateral_inhibition_strength': 0.05,
            'spontaneous_activity': 0.01
        },
        'temporal': {
            'membrane_time_constant': 0.01,
            'spike_threshold': -50.0,
            'resting_potential': -70.0
        },
        'motor_gains': {
            'forward_speed': 20.0,
            'rotation_speed': 45.0
        }
    }
    return config


def main():
    """Run a simple demo simulation."""
    
    logger.info("=" * 70)
    logger.info("NeuroMechFly Demo - Embodied Fly Simulation")
    logger.info("=" * 70)
    
    # Create configuration
    config = create_demo_config()
    logger.info("Configuration created")
    
    # Initialize components
    logger.info("Initializing simulation components...")
    
    # Arena
    arena = Arena(
        width=100, height=100, depth=50,
        food_position=[50, 50, 0],
        food_intensity=1.0
    )
    logger.info("✓ Arena initialized (100x100x50 mm)")
    
    # Brain
    brain = OlfactoryCircuit(config)
    logger.info(f"✓ Brain initialized ({config['neurons']['dn_count']} descending neurons)")
    
    # Body
    fly = FlyInterface(config)
    logger.info("✓ Fly body initialized (mock mode)")
    
    # Main simulation
    sim = NeuroMechFlySimulation(fly, brain, arena, config)
    logger.info("✓ Simulation ready")
    
    # Run short demo (5 seconds)
    logger.info("\n" + "-" * 70)
    logger.info("Running 5-second demo simulation...")
    logger.info("-" * 70 + "\n")
    
    num_steps = 5000  # 5 seconds (0.001s timestep)
    
    for step in range(num_steps):
        sim.step()
        
        if (step + 1) % 1000 == 0:
            pos = fly.position
            odor = arena.get_odor_concentration(pos)
            logger.info(f"Step {step + 1}/{num_steps} | "
                       f"Position: ({pos[0]:.1f}, {pos[1]:.1f}) | "
                       f"Odor: {odor:.3f}")
    
    # Print final statistics
    logger.info("\n" + "-" * 70)
    logger.info("DEMO SUMMARY")
    logger.info("-" * 70)
    
    positions = np.array(sim.logged_data['position'])
    
    if len(positions) > 0:
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        final_pos = positions[-1]
        final_odor = arena.get_odor_concentration(final_pos)
        food_pos = np.array([50, 50])
        dist_to_food = np.linalg.norm(final_pos[:2] - food_pos)
        
        logger.info(f"Total distance traveled: {total_distance:.1f} mm")
        logger.info(f"Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        logger.info(f"Final odor concentration: {final_odor:.4f}")
        logger.info(f"Distance to food source: {dist_to_food:.1f} mm")
    
    logger.info("\n✓ Demo complete!")
    logger.info("\nNext steps:")
    logger.info("  1. Run full simulation: python run_experiment.py --duration 60")
    logger.info("  2. Check the data/ folder for results")
    logger.info("  3. Edit config/ files to modify behavior")
    logger.info("  4. See README.md for more information")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
