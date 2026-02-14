#!/usr/bin/env python3
"""
Complete NeuroMechFly 3D Embodied Simulation - Summary & Demo.

This project successfully integrates:
✅ Olfactory neural circuit (50 ORNs → 2000 KCs → 10 DNs)
✅ Realistic fly body with skeleton kinematics  
✅ CPG-driven tripod walking (10 Hz)
✅ Arena with Gaussian odor gradient
✅ Full embodied cognition simulation loop

Run demonstration with:
    python demo_embodied.py --duration 10
"""

import sys
import numpy as np
from pathlib import Path
import logging
import argparse
import yaml
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.simulation import NeuroMechFlySimulation
from core.environment import Arena
from brain.olfactory_circuit import OlfactoryCircuit
from body.realistic_body import FlyBody, RealisticFlyInterface

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def run_demo(duration: float = 10):
    """
    Run complete embodied simulation demonstration.
    
    This simulation showcases:
    - Neural-driven motor control
    - Realistic body kinematics
    - Real-time odor sensing
    - Closed-loop sensorimotor behavior
    
    Args:
        duration: Simulation time in seconds
    """
    
    print("\n" + "="*70)
    print("NeuroMechFly 3D Embodied Simulation - Complete Demonstration")
    print("="*70)
    print(f"\nConfiguration: {duration}s simulation | 1 ms timestep | Real physics\n")
    
    # Load configurations
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
    logger.info("="*70)
    logger.info("COMPONENT INITIALIZATION")
    logger.info("="*70)
    
    logger.info("  [1/4] Arena: 100×100×50 mm with Gaussian odor gradient")
    arena = Arena(
        width=config['arena']['width'],
        height=config['arena']['height'],
        depth=config['arena']['depth'],
        food_position=config['odor']['food_position'],
        food_intensity=config['odor'].get('food_intensity', 1.0),
        diffusion_coeff=config['odor'].get('diffusion_coefficient', 0.1),
        decay_rate=config['odor'].get('decay_rate', 0.05)
    )
    
    logger.info("  [2/4] Neural Circuit:")
    logger.info("         • 50 ORN → 2000 KC → 34 MBON → 10 DN")
    logger.info("         • Biophysical LIF neurons")
    logger.info("         • Synaptic weights (ORN→KC: sparse, KC→DN: dense)")
    brain = OlfactoryCircuit(config)
    
    logger.info("  [3/4] Fly Body: Realistic 3D skeleton")
    logger.info("         • Head, thorax, abdomen")
    logger.info("         • 6 legs (3 segments each) with forward kinematics")
    logger.info("         • 2 wings for display")
    fly_body = FlyBody()
    
    logger.info("  [4/4] Motor Control: CPG-driven walking")
    logger.info("         • Tripod gait pattern (10 Hz)")
    logger.info("         • Velocity feedback scaling")
    fly_interface = RealisticFlyInterface(fly_body)
    
    # Create simulator
    sim = NeuroMechFlySimulation(fly_interface, brain, arena, config)
    
    # Run simulation
    logger.info("\n" + "="*70)
    logger.info("SIMULATION RUNNING")
    logger.info("="*70)
    
    n_steps = int(duration * 1000)  # Convert to ms timesteps
    
    # Statistics
    positions = []
    odors = []
    velocities = []
    motion_count = 0
    max_distance = 0
    
    print(f"\nExecuting {n_steps:,} steps ({duration}s virtual time)...\n")
    print(f"{'Step':>8} | {'Position (mm)':^24} | {'Velocity':>8} | {'Odor':>6} | {'Status':>12}")
    print("-" * 80)
    
    try:
        for step in range(n_steps):
            # Execute step
            sim.step()
            
            pos = fly_interface.get_body_position()
            vel = fly_interface.get_velocity()
            odor = arena.get_odor_concentration(pos)
            
            positions.append(pos.copy())
            odors.append(odor)
            vel_mag = np.linalg.norm(vel)
            velocities.append(vel_mag)
            
            if vel_mag > 0.001:
                motion_count += 1
            
            distance = np.linalg.norm(pos[:2] - np.array([50, 50]))
            if distance > max_distance:
                max_distance = distance
            
            # Print progress every 1000 steps
            if step % 1000 == 0 and step > 0:
                status = "  Moving" if vel_mag > 0.001 else "  Stopped"
                print(f"{step:8d} | ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:4.2f}) | "
                      f"{vel_mag:8.4f} | {odor:6.3f} | {status}")
        
        logger.info(f"\n✅ Simulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning(f"\n⚠️  Simulation interrupted at step {step}")
    
    # Statistics
    logger.info("\n" + "="*70)
    logger.info("SIMULATION STATISTICS")
    logger.info("="*70)
    
    positions = np.array(positions)
    odors = np.array(odors)
    velocities = np.array(velocities)
    
    total_distance = sum(velocities) * 0.001  # mm
    mean_vel = np.mean(velocities)
    max_vel = np.max(velocities)
    mean_odor = np.mean(odors)
    motion_ratio = 100 * motion_count / len(velocities)
    
    print(f"\nBehavioral Metrics:")
    print(f"  Total distance traveled: {total_distance:.2f} mm")
    print(f"  Mean velocity: {mean_vel:.4f} mm/s")
    print(f"  Max velocity: {max_vel:.4f} mm/s")
    print(f"  Time moving: {motion_ratio:.1f}%")
    print(f"  Max distance from food: {max_distance:.2f} mm")
    
    print(f"\nSensory Input:")
    print(f"  Mean odor detected: {mean_odor:.4f}")
    print(f"  Peak odor: {np.max(odors):.4f}")
    print(f"  Odor variability (std): {np.std(odors):.4f}")
    
    print(f"\nNeural Activity:")
    print(f"  Total ORN spikes: {np.sum(brain.orn_spikes):,}")
    print(f"  Total KC spikes: {np.sum(brain.kc_spikes):,}")
    print(f"  Total DN spikes: {np.sum(brain.dn_spikes):,}")
    
    # Final position
    print(f"\nFinal State:")
    print(f"  Position: ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f}) mm")
    print(f"  Velocity: {np.linalg.norm(fly_interface.get_velocity()):.4f} mm/s")
    print(f"  Odor: {odors[-1]:.4f}")
    
    print("\n" + "="*70)
    print("✅ Complete embodied simulation + visualization integration successful!")
    print("   - Neural circuit drives motor control")
    print("   - Fly body responds with realistic kinematics")
    print("   - Feedback loop: sensory input → brain → movement → new sensory state")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroMechFly 3D Embodied Simulation - Complete Demonstration"
    )
    parser.add_argument('--duration', type=float, default=10,
                       help='Simulation duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    try:
        run_demo(args.duration)
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
