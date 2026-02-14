"""
Analysis and visualization utilities for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def plot_trajectory(logged_data: Dict, output_file: str = None):
    """
    Plot the fly's trajectory in the arena.
    
    Args:
        logged_data: Dictionary with logged simulation data
        output_file: Optional file to save figure
    """
    positions = np.array(logged_data.get('position', []))
    
    if positions.size == 0:
        logger.warning("No position data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Trajectory')
    
    # Mark start and end
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
    
    # Add food source (assumed at center)
    ax.plot(50, 50, 'y*', markersize=20, label='Food')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Fly Trajectory')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        logger.info(f"Trajectory plot saved to {output_file}")
    
    return fig, ax


def plot_neural_activity(logged_data: Dict, output_file: str = None):
    """
    Plot neural activity over time.
    
    Args:
        logged_data: Dictionary with logged simulation data
        output_file: Optional file to save figure
    """
    spikes = logged_data.get('neural_spikes', [])
    
    if not spikes or len(spikes) == 0:
        logger.warning("No spike data to plot")
        return
    
    # Extract DN spikes (assuming these are in the spike dict)
    dn_spikes = np.array([s['dn'] for s in spikes if 'dn' in s])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raster plot of DN spikes
    for neuron_id in range(min(10, dn_spikes.shape[1])):
        spike_times = np.where(dn_spikes[:, neuron_id])[0]
        ax.vlines(spike_times, neuron_id - 0.4, neuron_id + 0.4, color='black', linewidth=0.5)
    
    ax.set_xlabel('Time step')
    ax.set_ylabel('DN neuron ID')
    ax.set_title('Descending Neuron Activity')
    ax.set_ylim(-1, 10)
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        logger.info(f"Neural activity plot saved to {output_file}")
    
    return fig, ax


def plot_odor_response(logged_data: Dict, output_file: str = None):
    """
    Plot odor concentration over time.
    
    Args:
        logged_data: Dictionary with logged simulation data
        output_file: Optional file to save figure
    """
    times = np.array(logged_data.get('time', []))
    odors = np.array(logged_data.get('odor_input', []))
    
    if odors.size == 0:
        logger.warning("No odor data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, odors, 'g-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Odor concentration')
    ax.set_title('Olfactory Input Over Time')
    ax.grid(True)
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        logger.info(f"Odor response plot saved to {output_file}")
    
    return fig, ax
