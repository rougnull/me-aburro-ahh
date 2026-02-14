"""
Integration adapter between DMN framework and NeuroMechFly embodied simulation.

This module connects the differentiable neural circuit (PyTorch) with the
existing NeuroMechFly physics simulation for end-to-end learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DMNtoEnvironment:
    """
    Adapter that converts DMN neural outputs to motor commands
    for the embodied simulation environment.
    """
    
    def __init__(self, env, scale_forward: float = 0.1, scale_turn: float = 0.05):
        """
        Initialize adapter.
        
        Args:
            env: NeuroMechFlySimulation environment
            scale_forward: Scaling for forward velocity
            scale_turn: Scaling for turning velocity
        """
        self.env = env
        self.scale_forward = scale_forward
        self.scale_turn = scale_turn
    
    def dn_output_to_motor_command(self, dn_spikes: torch.Tensor) -> np.ndarray:
        """
        Convert DN output spikes to motor commands.
        
        DN organization (simplified 10-neuron model):
        - DN[0:2]: Forward motion command
        - DN[2]: Backward motion command
        - DN[3:5]: Turning left/right
        - DN[5:10]: Reserved for future behaviors
        
        Args:
            dn_spikes: DN spike output (10,)
            
        Returns:
            Motor command array suitable for environment
        """
        dn_spikes = dn_spikes.detach().cpu().numpy()
        
        # Extract motor commands
        forward = (dn_spikes[0] + dn_spikes[1]) * self.scale_forward
        backward = dn_spikes[2] * self.scale_forward
        turning = (dn_spikes[3] - dn_spikes[4]) * self.scale_turn
        
        # Net forward velocity
        velocity_forward = forward - backward
        
        # Create motor command
        motor_cmd = np.array([
            velocity_forward,  # Forward/backward
            turning,          # Left/right turning
            0.0              # Vertical (not used in 2D simulation)
        ])
        
        return motor_cmd
    
    def get_odor_observation(self) -> torch.Tensor:
        """Get current odor observation as tensor."""
        odor = self.env.get_odor_at(self.env.fly_position)
        return torch.tensor(odor, dtype=torch.float32)
    
    def get_position(self) -> torch.Tensor:
        """Get current fly position as tensor."""
        pos = np.array(self.env.fly_position)
        return torch.tensor(pos, dtype=torch.float32)


class EmbodiedDMNSimulator:
    """
    Complete embodied simulation combining differentiable neural circuit
    with NeuroMechFly physics and environment.
    """
    
    def __init__(
        self,
        neural_circuit: nn.Module,
        environment,
        adapter: Optional[DMNtoEnvironment] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize embodied simulator.
        
        Args:
            neural_circuit: DifferentiableOlfactoryCircuit model
            environment: NeuroMechFlySimulation environment
            adapter: DMNtoEnvironment adapter (created if None)
            device: torch device
        """
        self.circuit = neural_circuit
        self.env = environment
        self.device = device or torch.device('cpu')
        self.adapter = adapter or DMNtoEnvironment(environment)
        
        # Move circuit to device
        self.circuit.to(self.device)
        
        # Trajectory storage
        self.trajectory = {
            'positions': [],
            'velocities': [],
            'odor_observations': [],
            'dn_outputs': [],
            'activations': {}
        }
        
        logger.info("EmbodiedDMNSimulator initialized")
    
    def reset(self):
        """Reset simulator for new episode."""
        self.env.reset()
        self.circuit.reset_state()
        self.trajectory = {
            'positions': [],
            'velocities': [],
            'odor_observations': [],
            'dn_outputs': [],
            'activations': {}
        }
    
    def step(self) -> Dict:
        """
        Execute one simulation step:
        1. Get odor observation
        2. Forward pass through neural circuit
        3. Convert DN output to motor command
        4. Step environment
        5. Record trajectory
        
        Returns:
            Dict with step information
        """
        # Get observation
        pos_before = self.adapter.get_position().cpu().numpy().copy()
        odor = self.adapter.get_odor_observation()
        
        # Neural forward pass
        dn_spikes, activations = self.circuit(odor, return_activations=True)
        
        # Convert to motor command and step environment
        motor_cmd = self.adapter.dn_output_to_motor_command(dn_spikes)
        self.env.step(motor_cmd)
        
        # Get new position
        pos_after = self.adapter.get_position().cpu().numpy()
        velocity = pos_after - pos_before
        
        # Record trajectory
        self.trajectory['positions'].append(pos_after.copy())
        self.trajectory['velocities'].append(velocity.copy())
        self.trajectory['odor_observations'].append(odor.item())
        self.trajectory['dn_outputs'].append(dn_spikes.detach().cpu().numpy().copy())
        
        # Store first step activations structure
        if not self.trajectory['activations']:
            for layer in activations.keys():
                self.trajectory['activations'][layer] = []
        
        for layer, act in activations.items():
            self.trajectory['activations'][layer].append(
                act.detach().cpu().numpy().copy()
            )
        
        return {
            'position': pos_after,
            'velocity': velocity,
            'odor': odor.item(),
            'dn_spikes': dn_spikes.detach().cpu().numpy(),
        }
    
    def rollout_episode(self, episode_length: int = 5000) -> Dict:
        """
        Execute complete episode and return trajectory.
        
        Args:
            episode_length: Number of timesteps
            
        Returns:
            Dict with trajectory data
        """
        self.reset()
        
        for step in range(episode_length):
            self.step()
        
        # Convert lists to tensors
        result = {
            'positions': torch.tensor(self.trajectory['positions'], dtype=torch.float32),
            'velocities': torch.tensor(self.trajectory['velocities'], dtype=torch.float32),
            'spikes': {
                layer: torch.tensor(acts, dtype=torch.float32)
                for layer, acts in self.trajectory['activations'].items()
            }
        }
        
        return result
    
    def get_trajectory_metrics(self, goal_position: Tuple[float, float, float]) -> Dict:
        """Compute metrics from current trajectory."""
        if not self.trajectory['positions']:
            return {}
        
        positions = np.array(self.trajectory['positions'])
        goal = np.array(goal_position)
        
        distances = np.linalg.norm(positions - goal, axis=1)
        
        metrics = {
            'final_distance': distances[-1],
            'mean_distance': distances.mean(),
            'min_distance': distances.min(),
            'max_distance': distances.max(),
            'total_distance': np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)),
        }
        
        return metrics


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from core.environment import Arena
    from core.simulation import NeuroMechFlySimulation
    from brain.olfactory_circuit import OlfactoryCircuit
    from body.realistic_body import RealisticFlyInterface
    from config.config_loader import load_config
    from simulation.mechanism import DifferentiableOlfactoryCircuit
    from connectome.fetch_data import FlyWireConnectome
    from connectome.adjacency_matrix import AdjacencyMatrixGenerator
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing DMN Embodied Integration")
    print("="*60)
    
    # Setup environment
    arena = Arena()
    config = load_config('config/default_config.yaml')
    
    interface = RealisticFlyInterface()
    brain = OlfactoryCircuit(config)
    
    sim = NeuroMechFlySimulation(interface, brain, arena, config)
    
    # Create DMN circuit
    fetcher = FlyWireConnectome()
    connectome = fetcher.fetch_olfactory_circuit(use_synthetic=True)
    
    generator = AdjacencyMatrixGenerator()
    weights_orn_pn, mask_orn_pn = generator.create_weight_matrix(connectome)
    weights_pn_kc, mask_pn_kc = generator.create_weight_matrix(connectome)
    weights_kc_mbon, mask_kc_mbon = generator.create_weight_matrix(connectome)
    weights_mbon_dn, mask_mbon_dn = generator.create_weight_matrix(connectome)
    
    masks = {
        'mask_orn_pn': mask_orn_pn,
        'mask_pn_kc': mask_pn_kc,
        'mask_kc_mbon': mask_kc_mbon,
        'mask_mbon_dn': mask_mbon_dn,
    }
    
    dmn_circuit = DifferentiableOlfactoryCircuit(
        connectivity_data=connectome,
        weights_orn_pn=weights_orn_pn,
        weights_pn_kc=weights_pn_kc,
        weights_kc_mbon=weights_kc_mbon,
        weights_mbon_dn=weights_mbon_dn,
        masks=masks,
        learnable=True
    )
    
    # Create embodied simulator
    embodied_sim = EmbodiedDMNSimulator(dmn_circuit, sim)
    
    print("\nRunning 100-step rollout...")
    embodied_sim.rollout_episode(episode_length=100)
    
    metrics = embodied_sim.get_trajectory_metrics((50, 50, 0))
    print(f"\nTrajectory metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
