"""
BPTT Training Loop for Differentiable Mechanical Networks (DMN)

Implements backpropagation through time for end-to-end learning of:
- Synaptic weights
- Neuron time constants
- Motor control parameters

Used to optimize neural circuits for specific behavioral tasks
(e.g., chemotaxis, obstacle avoidance, energy efficiency).

Author: NeuroMechFly DMN Framework
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for DMN training."""
    # Training parameters
    learning_rate: float = 0.001
    momentum: float = 0.9
    optimizer: str = 'adam'  # 'adam', 'sgd', 'rmsprop'
    
    # Training schedule
    num_episodes: int = 100
    episode_length_steps: int = 5000  # 5 seconds at 1ms timestep
    batch_size: int = 4  # Episodes per batch
    
    # Loss weights
    nav_weight: float = 1.0
    energy_weight: float = 0.1
    sparsity_weight: float = 0.01
    activity_weight: float = 0.001
    
    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'step'  # 'step' or 'reduce_on_plateau'
    lr_decay_steps: int = 50
    lr_decay_factor: float = 0.5
    
    # Checkpoint & logging
    save_interval: int = 10
    checkpoint_dir: str = 'checkpoints'
    log_interval: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class DMNTrainer:
    """
    Trainer for Differentiable Mechanical Networks using BPTT.
    
    Orchestrates:
    - Forward pass through neural circuit over full episode
    - Loss computation with all regularization terms
    - Backward pass (automatic differentiation through time)
    - Parameter updates via optimization
    """
    
    def __init__(
        self,
        network: nn.Module,
        loss_function: nn.Module,
        config: TrainingConfig,
        environment=None
    ):
        """
        Initialize DMN trainer.
        
        Args:
            network: Neural circuit model (DifferentiableOlfactoryCircuit)
            loss_function: Loss function (CombinedLoss)
            config: Training configuration
            environment: Simulation environment for rollouts
        """
        self.network = network
        self.loss_fn = loss_function
        self.config = config
        self.environment = environment
        
        self.device = torch.device(config.device)
        self.network.to(self.device)
        self.loss_fn.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer(config)
        self.scheduler = self._setup_scheduler(config)
        
        # Training state
        self.global_step = 0
        self.episode = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"DMNTrainer initialized on {self.device}")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        logger.info(f"Learnable parameters: {sum(p.numel() for p in self.network.parameters() if p.requires_grad):,}")
    
    def _setup_optimizer(self, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer based on config."""
        learnable_params = [p for p in self.network.parameters() if p.requires_grad]
        
        if config.optimizer == 'adam':
            return optim.Adam(learnable_params, lr=config.learning_rate, betas=(0.9, 0.999))
        elif config.optimizer == 'sgd':
            return optim.SGD(learnable_params, lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer == 'rmsprop':
            return optim.RMSprop(learnable_params, lr=config.learning_rate, momentum=config.momentum)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _setup_scheduler(self, config: TrainingConfig):
        """Setup learning rate scheduler."""
        if not config.use_scheduler:
            return None
        
        if config.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=config.lr_decay_steps,
                gamma=config.lr_decay_factor
            )
        elif config.scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_decay_factor,
                patience=5,
                verbose=True
            )
        return None
    
    def rollout_episode(
        self,
        episode_length: int,
        goal_position: Tuple[float, float, float] = (50.0, 50.0, 0.0)
    ) -> Dict[str, torch.Tensor]:
        """
        Execute one complete episode rollout with current network.
        
        Returns trajectories and neural activity for loss computation.
        
        Args:
            episode_length: Number of timesteps
            goal_position: Task goal
            
        Returns:
            Dict with 'positions', 'velocities', 'spikes', 'observations'
        """
        if self.environment is None:
            # Generate synthetic trajectory for testing
            return self._synthetic_rollout(episode_length, goal_position)
        
        # Reset for new episode
        self.network.reset_state()
        self.environment.reset()
        
        # Trajectory storage
        positions = []
        velocities = []
        observations = []
        spike_dict = {layer: [] for layer in ['orn', 'pn', 'kc', 'mbon', 'dn']}
        
        current_pos = torch.tensor(self.environment.fly_position, dtype=torch.float32, device=self.device)
        
        # Rollout
        for step in range(episode_length):
            # Get observation (odor concentration)
            odor = self.environment.get_odor_at(self.environment.fly_position)
            odor_tensor = torch.tensor([[odor]], dtype=torch.float32, device=self.device)
            
            # Forward pass through network
            dn_spikes, activations = self.network(odor_tensor, return_activations=True)
            
            # Convert DN output to motor commands
            motor_commands = self._dn_to_motor(dn_spikes)
            
            # Step environment
            self.environment.step(motor_commands.cpu().numpy())
            
            # Record trajectory
            new_pos = torch.tensor(self.environment.fly_position, dtype=torch.float32, device=self.device)
            velocity = new_pos - current_pos
            
            positions.append(current_pos)
            velocities.append(velocity)
            observations.append(odor_tensor)
            
            for layer in ['orn', 'pn', 'kc', 'mbon', 'dn']:
                spike_dict[layer].append(activations[layer])
            
            current_pos = new_pos
        
        # Stack trajectories
        result = {
            'positions': torch.stack(positions),  # (T, 3)
            'velocities': torch.stack(velocities),  # (T, 3)
            'observations': torch.cat(observations),  # (T, 1)
            'spikes': {k: torch.stack(v) for k, v in spike_dict.items()},  # (T, n_neurons)
        }
        
        return result
    
    def _synthetic_rollout(
        self,
        episode_length: int,
        goal_position: Tuple[float, float, float]
    ) -> Dict[str, torch.Tensor]:
        """Generate synthetic rollout for testing (no environment)."""
        # Simulated network rollout
        self.network.reset_state()
        
        positions = []
        velocities = []
        spike_dict = {layer: [] for layer in ['orn', 'pn', 'kc', 'mbon', 'dn']}
        
        current_pos = torch.tensor([25.0, 25.0, 0.0], dtype=torch.float32, device=self.device)
        
        for step in range(episode_length):
            # Synthetic odor gradient (higher near goal)
            dist_to_goal = torch.norm(current_pos - torch.tensor(goal_position, device=self.device))
            odor = torch.exp(-dist_to_goal / 20.0).unsqueeze(0).unsqueeze(0)
            
            # Network forward pass
            dn_spikes, activations = self.network(odor.squeeze(), return_activations=True)
            
            # Convert DN to velocity (simple linear mapping)
            velocity = dn_spikes[:3].unsqueeze(0) * 0.01
            if len(dn_spikes) > 3:
                velocity[:, 0] += dn_spikes[3].unsqueeze(0) * -0.01  # DN3 for turning
            
            # Update position
            new_pos = current_pos + velocity.squeeze()
            
            # Record
            positions.append(current_pos)
            velocities.append(velocity.squeeze())
            
            for layer in ['orn', 'pn', 'kc', 'mbon', 'dn']:
                spike_dict[layer].append(activations[layer])
            
            current_pos = new_pos
        
        return {
            'positions': torch.stack(positions),
            'velocities': torch.stack(velocities),
            'spikes': {k: torch.stack(v) for k, v in spike_dict.items()},
        }
    
    def _dn_to_motor(self, dn_spikes: torch.Tensor) -> torch.Tensor:
        """Convert DN spike output to motor commands (velocity)."""
        # Simple linear mapping: DN0-1 → forward, DN2 → backward, DN3-4 → turning
        if len(dn_spikes) >= 4:
            forward = dn_spikes[0] + dn_spikes[1]
            backward = dn_spikes[2]
            turning = dn_spikes[3] - dn_spikes[4] if len(dn_spikes) > 4 else dn_spikes[3]
        else:
            forward = dn_spikes[0] if len(dn_spikes) > 0 else 0
            backward = dn_spikes[1] if len(dn_spikes) > 1 else 0
            turning = dn_spikes[2] if len(dn_spikes) > 2 else 0
        
        # Combine into 3D motor command
        motor_cmd = torch.stack([
            (forward - backward) * 0.1,  # Forward/backward
            turning * 0.05,            # Turning
            torch.zeros_like(forward)  # Z (no vertical movement)
        ])
        
        return motor_cmd
    
    def train_step(self, episode_length: int) -> Dict[str, float]:
        """
        Execute one training step: rollout → loss → backward → update.
        
        Args:
            episode_length: Number of timesteps per episode
            
        Returns:
            Dict with loss components
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        # Rollout episode
        trajectory = self.rollout_episode(episode_length)
        
        # Compute loss
        loss_dict = self.loss_fn(
            positions=trajectory['positions'],
            velocities=trajectory['velocities'],
            spikes=trajectory['spikes']
        )
        
        total_loss = loss_dict['total']
        
        # Backward pass (BPTT)
        total_loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Convert loss dict to scalars
        scalar_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        
        self.global_step += 1
        self.loss_history.append(scalar_losses['total'])
        
        return scalar_losses
    
    def train_episode(self, num_episodes: int) -> List[Dict]:
        """
        Train for multiple episodes.
        
        Args:
            num_episodes: Number of episodes to train
            
        Returns:
            List of loss dictionaries per episode
        """
        episode_losses = []
        
        logger.info(f"Starting training for {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            self.episode = ep
            
            # Training step
            losses = self.train_step(self.config.episode_length_steps)
            episode_losses.append(losses)
            
            # Logging
            if (ep + 1) % self.config.log_interval == 0:
                logger.info(f"Episode {ep+1}/{num_episodes} | "
                           f"Loss: {losses['total']:.4f} | "
                           f"Nav: {losses['nav_distance']:.4f} | "
                           f"Energy: {losses.get('energy', 0):.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(losses['total'])
                else:
                    self.scheduler.step()
            
            # Checkpointing
            if (ep + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"episode_{ep+1}.pt")
                
                # Save best model
                if losses['total'] < self.best_loss:
                    self.best_loss = losses['total']
                    self.save_checkpoint("best_model.pt")
        
        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")
        
        return episode_losses
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'episode': self.episode,
            'global_step': self.global_step,
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'best_loss': self.best_loss,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.episode = checkpoint['episode']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from episode {self.episode}, best_loss={self.best_loss:.4f}")
    
    def get_loss_history(self) -> List[float]:
        """Get training loss history."""
        return self.loss_history


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("DMN Training Framework Demo")
    print("="*60)
    
    # Would load real network and training components here
    # For now, just show the structure
    
    config = TrainingConfig(
        num_episodes=100,
        episode_length_steps=5000,
        learning_rate=0.001,
        optimizer='adam'
    )
    
    print(f"\nTraining Configuration:")
    for key, val in asdict(config).items():
        print(f"  {key}: {val}")
    
    print(f"\nTo use the trainer:")
    print(f"  1. Create DifferentiableOlfactoryCircuit network")
    print(f"  2. Create CombinedLoss function")
    print(f"  3. Initialize DMNTrainer(network, loss_fn, config)")
    print(f"  4. Call trainer.train_episode(num_episodes)")
    
    print(f"\nExample pseudocode:")
    print("""
    from connectome.fetch_data import FlyWireConnectome
    from connectome.adjacency_matrix import AdjacencyMatrixGenerator
    from simulation.mechanism import DifferentiableOlfactoryCircuit
    from training.loss_functions import CombinedLoss
    from training.optimizer import DMNTrainer, TrainingConfig
    
    # Setup
    fetcher = FlyWireConnectome()
    connectome = fetcher.fetch_olfactory_circuit()
    generator = AdjacencyMatrixGenerator()
    weights_orn_pn, mask_orn_pn = generator.create_weight_matrix(...)
    
    # Create network
    circuit = DifferentiableOlfactoryCircuit(
        connectivity_data=connectome,
        weights_orn_pn=weights_orn_pn,
        ...
        learnable=True
    )
    
    # Setup training
    loss_fn = CombinedLoss(navigation_weight=1.0, energy_weight=0.1)
    config = TrainingConfig(num_episodes=100)
    trainer = DMNTrainer(circuit, loss_fn, config)
    
    # Train
    losses = trainer.train_episode(num_episodes=100)
    trainer.save_checkpoint("final_model.pt")
    """)
