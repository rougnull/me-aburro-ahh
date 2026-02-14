"""
Main simulation loop orchestrating the integration between:
- Physical simulation (MuJoCo via NeuroMechFly)
- Neural model (Brain)
- Motor control (Body)
- Environment (Odor, wind, etc.)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class NeuroMechFlySimulation:
    """
    Main simulation class that orchestrates the integration between
    neural model, physical body, and environment.
    """
    
    def __init__(self, fly_interface, brain_model, environment, config):
        """
        Initialize the simulation.
        
        Args:
            fly_interface: NeuroMechFly robot interface
            brain_model: Neural circuit model
            environment: Arena and odor environment
            config: Configuration dictionary
        """
        self.fly = fly_interface
        self.brain = brain_model
        self.env = environment
        self.config = config
        
        self.timestep = config['physics']['timestep']
        self.current_time = 0.0
        self.step_count = 0
        
        # Data logging
        self.logged_data = {
            'time': [],
            'position': [],
            'orientation': [],
            'velocity': [],
            'odor_input': [],
            'brain_output': [],
            'motor_commands': [],
            'neural_spikes': []
        }
        
        logger.info("NeuroMechFly Simulation initialized")
    
    def step(self) -> Dict:
        """
        Execute one simulation step:
        1. Read physical state
        2. Calculate sensory input (odor)
        3. Process neural model
        4. Convert to motor commands
        5. Update physics
        6. Log data
        
        Returns:
            Dictionary with step information
        """
        
        # 1. Get current physical state
        pos = self.fly.get_body_position()
        orientation = self.fly.get_orientation()
        velocity = self.fly.get_velocity()
        
        # 2. Calculate odor concentration at fly position
        odor_input = self.env.get_odor_concentration(pos)
        
        # 3. Process brain (advance neural simulation)
        brain_output, spikes = self.brain.step(odor_input, self.timestep)
        
        # 4. Convert neural output to motor commands
        # brain_output typically contains descending neuron activity
        motor_commands = self._decode_motor_commands(brain_output)
        
        # 5. Apply commands and update physics
        self.fly.apply_motor_commands(motor_commands)
        self.fly.physics_step(self.timestep)
        
        # 6. Log data
        self._log_data(pos, orientation, velocity, odor_input, 
                      brain_output, motor_commands, spikes)
        
        self.current_time += self.timestep
        self.step_count += 1
        
        return {
            'time': self.current_time,
            'position': pos,
            'odor_input': odor_input,
            'motor_commands': motor_commands
        }
    
    def _decode_motor_commands(self, brain_output: np.ndarray) -> Dict[str, float]:
        """
        Decode descending neuron activity to motor commands.
        Assumes brain_output is normalized to [-1, 1] range.
        
        Args:
            brain_output: Activity of descending neurons (normalized)
            
        Returns:
            Dictionary with forward_speed and angular_velocity
        """
        # Simple linear decoding
        num_dn = len(brain_output)
        
        # Split DN population
        if num_dn >= 8:
            # First half -> forward
            forward_dn = brain_output[:(num_dn//2)]
            # Second half -> turning
            turn_dn = brain_output[(num_dn//2):]
            
            forward_normalized = np.mean(forward_dn)
            turn_normalized = np.mean(turn_dn)
        else:
            forward_normalized = np.mean(brain_output)
            turn_normalized = 0.0
        
        # Clip and scale
        forward_normalized = np.clip(forward_normalized, -1, 1)
        turn_normalized = np.clip(turn_normalized, -1, 1)
        
        # Apply motor gains (in mm/s and deg/s)
        forward_speed = forward_normalized * \
                       self.config.get('motor_gains', {}).get('forward_speed', 20.0)
        
        rotation_speed = turn_normalized * \
                        self.config.get('motor_gains', {}).get('rotation_speed', 45.0)
        
        return {
            'forward_speed': float(forward_speed),
            'angular_velocity': float(rotation_speed)
        }
    
    def _log_data(self, pos, orientation, velocity, odor_input, 
                  brain_output, motor_commands, spikes):
        """Log simulation data for analysis."""
        self.logged_data['time'].append(self.current_time)
        self.logged_data['position'].append(pos.copy())
        self.logged_data['orientation'].append(orientation.copy())
        self.logged_data['velocity'].append(velocity.copy())
        self.logged_data['odor_input'].append(float(odor_input))
        self.logged_data['brain_output'].append(brain_output.copy())
        self.logged_data['motor_commands'].append(motor_commands)
        self.logged_data['neural_spikes'].append(spikes)
    
    def run(self, num_steps: int, verbose: bool = True):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of simulation steps
            verbose: Print progress
        """
        logger.info(f"Starting simulation for {num_steps} steps")
        
        for step in range(num_steps):
            self.step()
            
            if verbose and (step + 1) % 1000 == 0:
                logger.info(f"Completed step {step + 1}/{num_steps} "
                           f"(t={self.current_time:.2f}s)")
        
        logger.info("Simulation complete")
    
    def save_data(self, filepath: str):
        """Save logged data to HDF5 file with better serialization."""
        import h5py
        
        with h5py.File(filepath, 'w') as f:
            # Save numeric data directly
            for key in ['time', 'position', 'orientation', 'velocity', 'odor_input', 'brain_output']:
                data = self.logged_data.get(key, [])
                if data:
                    try:
                        f.create_dataset(key, data=np.array(data))
                    except Exception as e:
                        logger.warning(f"Could not save {key}: {e}")
            
            # Save motor commands as float32 arrays
            motor_cmd = self.logged_data.get('motor_commands', [])
            if motor_cmd:
                try:
                    forward = []
                    angular = []
                    for cmd in motor_cmd:
                        if isinstance(cmd, dict):
                            forward.append(cmd.get('forward_speed', 0.0))
                            angular.append(cmd.get('angular_velocity', 0.0))
                    if forward:
                        f.create_dataset('motor_forward', data=np.array(forward, dtype=np.float32))
                        f.create_dataset('motor_angular', data=np.array(angular, dtype=np.float32))
                except Exception as e:
                    logger.warning(f"Could not save motor commands: {e}")
            
            # Save spike data as spike counts per layer
            spikes = self.logged_data.get('neural_spikes', [])
            if spikes:
                try:
                    spike_counts = {'orn': [], 'pn': [], 'kc': [], 'mbon': [], 'dn': []}
                    for spike_dict in spikes:
                        if spike_dict:
                            for layer in spike_counts.keys():
                                if layer in spike_dict:
                                    spike_counts[layer].append(int(np.sum(spike_dict[layer])))
                                else:
                                    spike_counts[layer].append(0)
                    
                    # Save spike counts for each layer
                    for layer, counts in spike_counts.items():
                        if counts:
                            f.create_dataset(f'spike_count_{layer}', data=np.array(counts, dtype=np.int32))
                except Exception as e:
                    logger.warning(f"Could not save spike data: {e}")
        
        logger.info(f"Simulation data saved to {filepath}")
    
    def get_logged_data(self) -> Dict:
        """Return all logged data."""
        return self.logged_data
