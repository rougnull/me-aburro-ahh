#!/usr/bin/env python3
"""
Interactive viewer for NeuroMechFly simulation results.
Load and visualize HDF5 data, generate additional analyses, and display statistics.

Usage:
    python view_results.py data/20260214_225011/simulation_data.h5
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_simulation_data(h5_filepath):
    """Load HDF5 simulation data."""
    data = {}
    try:
        with h5py.File(h5_filepath, 'r') as f:
            logger.info(f"Available datasets in {h5_filepath}:")
            for key in f.keys():
                data[key] = np.array(f[key])
                logger.info(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
    except Exception as e:
        logger.error(f"Failed to load {h5_filepath}: {e}")
        return None
    
    return data


def print_statistics(data):
    """Print detailed statistics about the simulation."""
    print("\n" + "=" * 70)
    print("NEUROMECHFLY SIMULATION STATISTICS")
    print("=" * 70)
    
    if 'time' in data:
        times = data['time']
        print(f"\n⏱️  TEMPORAL:")
        print(f"   - Duration: {times[-1]:.2f} s")
        print(f"   - Timesteps: {len(times)}")
        print(f"   - Dt: {times[1] - times[0]:.4f} s")
    
    if 'position' in data:
        positions = data['position']
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        print(f"\n📍 MOVEMENT:")
        print(f"   - Total distance: {np.sum(distances):.1f} mm")
        print(f"   - Max distance per step: {np.max(distances):.4f} mm")
        print(f"   - Mean velocity: {np.mean(distances) * 1000:.2f} mm/s")
        print(f"   - Final position: ({positions[-1, 0]:.1f}, {positions[-1, 1]:.1f}, {positions[-1, 2]:.1f})")
        print(f"   - X range: {np.min(positions[:, 0]):.1f} - {np.max(positions[:, 0]):.1f} mm")
        print(f"   - Y range: {np.min(positions[:, 1]):.1f} - {np.max(positions[:, 1]):.1f} mm")
    
    if 'odor_input' in data:
        odors = data['odor_input']
        print(f"\n👃 OLFACTORY INPUT:")
        print(f"   - Min concentration: {np.min(odors):.4f}")
        print(f"   - Max concentration: {np.max(odors):.4f}")
        print(f"   - Mean concentration: {np.mean(odors):.4f}")
        print(f"   - Std deviation: {np.std(odors):.4f}")
    
    if 'brain_output' in data:
        brain = data['brain_output']
        print(f"\n🧠 NEURAL OUTPUT:")
        print(f"   - Output shape: {brain.shape}")
        print(f"   - Min activity: {np.min(brain):.2f}")
        print(f"   - Max activity: {np.max(brain):.2f}")
        print(f"   - Mean activity: {np.mean(brain):.2f}")
    
    if 'motor_forward' in data and 'motor_angular' in data:
        forward = data['motor_forward']
        angular = data['motor_angular']
        print(f"\n🚶 MOTOR COMMANDS:")
        print(f"   - Forward speed range: {np.min(forward):.2f} - {np.max(forward):.2f} mm/s")
        print(f"   - Angular velocity range: {np.min(angular):.2f} - {np.max(angular):.2f} deg/s")
        print(f"   - Mean forward: {np.mean(forward):.2f} mm/s")
        print(f"   - Mean angular: {np.mean(angular):.2f} deg/s")
    
    # Spike data
    spike_keys = [k for k in data.keys() if 'spike_count' in k]
    if spike_keys:
        print(f"\n📊 NEURAL SPIKES:")
        for key in spike_keys:
            spikes = data[key]
            layer_name = key.replace('spike_count_', '').upper()
            print(f"   - {layer_name}: {np.sum(spikes)} total spikes")
    
    print("\n" + "=" * 70)


def plot_quick_analysis(data, output_dir=None):
    """Create a quick multi-panel analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Position over time
    if 'time' in data and 'position' in data:
        times = data['time']
        positions = data['position']
        
        ax = axes[0, 0]
        ax.plot(times, positions[:, 0], 'b-', label='X', linewidth=1)
        ax.plot(times, positions[:, 1], 'r-', label='Y', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (mm)')
        ax.set_title('Position Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Velocity
    if 'time' in data and 'position' in data:
        times = data['time']
        positions = data['position']
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        ax = axes[0, 1]
        ax.plot(times[1:], velocities * 1000, 'g-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (mm/s)')
        ax.set_title('Instantaneous Velocity')
        ax.grid(True, alpha=0.3)
    
    # Odor perception
    if 'time' in data and 'odor_input' in data:
        times = data['time']
        odors = data['odor_input']
        
        ax = axes[1, 0]
        ax.fill_between(times, 0, odors, alpha=0.5, color='green')
        ax.plot(times, odors, 'g-', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Odor Concentration')
        ax.set_title('Olfactory Input')
        ax.grid(True, alpha=0.3)
    
    # Motor commands
    if 'motor_forward' in data:
        forward = data['motor_forward']
        ax = axes[1, 1]
        ax.plot(forward, 'b-', label='Forward', linewidth=0.8, alpha=0.7)
        if 'motor_angular' in data:
            angular = data['motor_angular']
            ax.plot(angular, 'r-', label='Angular', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Command Value')
        ax.set_title('Motor Commands')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'analysis_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Summary plot saved to {output_path}")
    
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description='View NeuroMechFly simulation results')
    parser.add_argument('filepath', help='Path to HDF5 simulation data file')
    parser.add_argument('--stats', action='store_true', help='Print detailed statistics')
    parser.add_argument('--plot', action='store_true', help='Generate analysis plots')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Directory to save generated plots')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading simulation data from {args.filepath}")
    data = load_simulation_data(args.filepath)
    
    if data is None:
        return 1
    
    # Print statistics
    if args.stats or True:  # Always print stats
        print_statistics(data)
    
    # Generate plots
    if args.plot:
        output_dir = args.output_dir or Path(args.filepath).parent
        logger.info(f"Generating analysis plots to {output_dir}")
        plot_quick_analysis(data, output_dir)
        plt.show()
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
