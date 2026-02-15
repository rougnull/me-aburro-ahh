#!/usr/bin/env python3
"""
Real NeuroMechFly DMN Training

Combines differentiable neural circuits with embodied fly simulation
for end-to-end learning of navigation behaviors.

Run: python train_embodied_dmn.py --num-episodes 20
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from pathlib import Path
import json

# Import NeuroMechFly components
from core.environment import Arena
from core.simulation import NeuroMechFlySimulation
from brain.olfactory_circuit import OlfactoryCircuit
from body.realistic_body import RealisticFlyInterface
from config.config_loader import load_default_config

# Import DMN components
from connectome.fetch_data import FlyWireConnectome
from connectome.adjacency_matrix import AdjacencyMatrixGenerator
from simulation.mechanism import DifferentiableOlfactoryCircuit
from training.loss_functions import CombinedLoss, TrajectoryAnalyzer
from dmn_embodied_integration import EmbodiedDMNSimulator, DMNtoEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_embodied_dmn(config: dict) -> tuple:
    """
    Setup embodied DMN simulator with real NeuroMechFly components.
    
    Returns:
        (embodied_simulator, loss_function, optimizer, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup NeuroMechFly environment
    logger.info("Initializing NeuroMechFly embodied simulation...")
    nmf_config = load_default_config()  # Load merged config from YAML files
    
    # Initialize arena with config
    env_params = nmf_config.get('arena', {})
    odor_params = nmf_config.get('odor', {})
    arena = Arena(
        width=env_params.get('width', 100.0),
        height=env_params.get('height', 100.0),
        depth=env_params.get('depth', 50.0),
        food_position=odor_params.get('food_position', [50, 50, 0]),
        food_intensity=odor_params.get('food_intensity', 1.0),
        diffusion_coeff=odor_params.get('diffusion_coefficient', 0.1),
        decay_rate=odor_params.get('decay_rate', 0.05)
    )
    
    interface = RealisticFlyInterface()
    brain = OlfactoryCircuit(nmf_config)
    
    sim = NeuroMechFlySimulation(interface, brain, arena, nmf_config)
    
    # Load connectome
    logger.info("Loading FlyWire connectome...")
    fetcher = FlyWireConnectome(use_cache=True)
    connectome = fetcher.fetch_olfactory_circuit(
        include_layers=['ORN', 'PN', 'KC', 'MBON', 'DN'],
        use_synthetic=True
    )
    
    logger.info(f"Connectome: {len(connectome['neurons'])} neurons, "
               f"{connectome['total_synapses']} synapses")
    
    # Create weight matrices with correct layer sizes
    logger.info("Generating weight matrices...")
    
    # Define layer sizes for olfactory circuit
    layer_sizes = {
        'orn': 50,
        'pn': 50,
        'kc': 2000,
        'mbon': 50,
        'dn': 10
    }
    
    layer_pairs = [('orn', 'pn'), ('pn', 'kc'), ('kc', 'mbon'), ('mbon', 'dn')]
    
    matrices = {}
    masks = {}
    
    # For each layer pair, create appropriately sized weight matrix
    # Weights are stored as (pre_size, post_size) for proper matrix multiply
    for pre_layer, post_layer in layer_pairs:
        pre_size = layer_sizes[pre_layer]
        post_size = layer_sizes[post_layer]
        
        # Create random weight matrix (will be learned)
        # Shape: (pre_size, post_size) so that .T @ pre_neurons gives post neurons
        w = torch.randn(pre_size, post_size) * 0.1
        w = w.to(device)
        
        # Create all-ones mask (full connectivity for learning)
        m = torch.ones(pre_size, post_size)
        m = m.to(device)
        
        matrices[(pre_layer, post_layer)] = (w, m)
        masks[f'mask_{pre_layer}_{post_layer}'] = m
    
    # Create DMN circuit  
    logger.info("Building differentiable neural circuit...")
    dmn_circuit = DifferentiableOlfactoryCircuit(
        connectivity_data=connectome,
        weights_orn_pn=matrices[('orn', 'pn')][0],
        weights_pn_kc=matrices[('pn', 'kc')][0],
        weights_kc_mbon=matrices[('kc', 'mbon')][0],
        weights_mbon_dn=matrices[('mbon', 'dn')][0],
        masks=masks,
        device=device,
        learnable=True
    )
    
    # Create embodied simulator
    logger.info("Creating embodied simulator...")
    adapter = DMNtoEnvironment(sim)
    embodied_sim = EmbodiedDMNSimulator(dmn_circuit, sim, adapter, device)
    
    # Loss function
    loss_fn = CombinedLoss(
        navigation_weight=config.get('nav_weight', 1.0),
        energy_weight=config.get('energy_weight', 0.1),
        sparsity_weight=config.get('sparse_weight', 0.01),
        activity_weight=config.get('activity_weight', 0.001),
        goal_position=(50.0, 50.0, 0.0)
    )
    loss_fn = loss_fn.to(device)
    
    # Optimizer
    learnable_params = [p for p in dmn_circuit.parameters() if p.requires_grad]
    optimizer = optim.Adam(learnable_params, lr=config.get('learning_rate', 0.001))
    
    logger.info(f"DMN setup complete!")
    logger.info(f"  Total parameters: {sum(p.numel() for p in dmn_circuit.parameters()):,}")
    logger.info(f"  Learnable parameters: {sum(p.numel() for p in learnable_params):,}")
    
    return embodied_sim, loss_fn, optimizer, device


def train_embodied_dmn(
    embodied_sim: EmbodiedDMNSimulator,
    loss_fn,
    optimizer,
    device,
    config: dict
):
    """
    Train DMN with embodied NeuroMechFly simulation.
    """
    num_episodes = config.get('num_episodes', 20)
    episode_length = config.get('episode_length', 5000)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Training")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Episode length: {episode_length} steps")
    logger.info(f"Goal: Navigate to (50, 50, 0)")
    
    training_losses = []
    
    for episode in range(num_episodes):
        embodied_sim.circuit.train()
        optimizer.zero_grad()
        
        # Rollout episode
        trajectory = embodied_sim.rollout_episode(episode_length=episode_length)
        
        # Compute loss
        losses = loss_fn(
            positions=trajectory['positions'],
            velocities=trajectory['velocities'],
            spikes=trajectory['spikes']
        )
        
        total_loss = losses['total']
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(embodied_sim.circuit.parameters(), max_norm=5.0)
        optimizer.step()
        
        # Log
        loss_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
        training_losses.append(loss_dict)
        
        if (episode + 1) % max(1, num_episodes // 10) == 0 or episode == 0:
            logger.info(f"Episode {episode+1:3d}/{num_episodes} | "
                       f"Loss: {loss_dict['total']:8.4f} | "
                       f"Nav: {loss_dict['nav_distance']:8.4f} | "
                       f"Energy: {loss_dict.get('energy', 0):8.4f}")
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best loss: {min(l['total'] for l in training_losses):.4f}")
    
    return training_losses


def evaluate_embodied_dmn(
    embodied_sim: EmbodiedDMNSimulator,
    device
):
    """
    Evaluate final trained network.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation")
    logger.info(f"{'='*60}")
    
    embodied_sim.circuit.eval()
    with torch.no_grad():
        trajectory = embodied_sim.rollout_episode(episode_length=5000)
        
        metrics = embodied_sim.get_trajectory_metrics((50, 50, 0))
        
        logger.info(f"\nFinal Trajectory Metrics:")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
    
    return trajectory, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train NeuroMechFly DMN with embodied simulation'
    )
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=5000,
                       help='Steps per episode')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--nav-weight', type=float, default=1.0)
    parser.add_argument('--energy-weight', type=float, default=0.1)
    
    args = parser.parse_args()
    
    config = {
        'num_episodes': args.num_episodes,
        'episode_length': args.episode_length,
        'learning_rate': args.learning_rate,
        'nav_weight': args.nav_weight,
        'energy_weight': args.energy_weight,
        'sparse_weight': 0.01,
        'activity_weight': 0.001,
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"NeuroMechFly DMN Training")
    logger.info(f"{'='*60}")
    
    try:
        # Setup
        embodied_sim, loss_fn, optimizer, device = setup_embodied_dmn(config)
        
        # Train
        training_losses = train_embodied_dmn(
            embodied_sim, loss_fn, optimizer, device, config
        )
        
        # Evaluate
        trajectory, metrics = evaluate_embodied_dmn(embodied_sim, device)
        
        # Save results
        results_dir = Path('embodied_training_results')
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'config': config,
            'training_losses': [
                {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                 for k, v in loss.items()}
                for loss in training_losses
            ],
            'final_metrics': {k: float(v) for k, v in metrics.items()}
        }
        
        with open(results_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {results_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
