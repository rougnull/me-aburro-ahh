"""
Complete DMN Training Example

Full end-to-end example showing:
1. Connectome data loading
2. Weight matrix generation
3. Differentiable circuit creation
4. Task-based training with BPTT
5. Results analysis and visualization

Run with:
    python dmn_train_example.py --num-episodes 20 --learning-rate 0.001
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Import DMN framework
from connectome.fetch_data import FlyWireConnectome
from connectome.adjacency_matrix import AdjacencyMatrixGenerator, ConnectivityConfig
from simulation.mechanism import DifferentiableOlfactoryCircuit
from training.loss_functions import CombinedLoss, TrajectoryAnalyzer
from training.optimizer import DMNTrainer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DMNExperiment:
    """Complete DMN training experiment."""
    
    def __init__(self, config: dict):
        """Initialize experiment with configuration."""
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        self.circuit = None
        self.trainer = None
        self.results = {
            'config': config,
            'training_losses': [],
            'final_metrics': {}
        }
    
    def setup_connectome(self) -> dict:
        """Load FlyWire connectome data."""
        logger.info("="*60)
        logger.info("STAGE 1: Loading Connectome")
        logger.info("="*60)
        
        fetcher = FlyWireConnectome(use_cache=True)
        
        connectome = fetcher.fetch_olfactory_circuit(
            include_layers=['ORN', 'PN', 'KC', 'MBON', 'DN'],
            use_synthetic=not self.config.get('use_real_flywire', False)
        )
        
        logger.info(f"Loaded connectome with {len(connectome['neurons'])} neurons")
        logger.info(f"Total synapses: {connectome['total_synapses']}")
        logger.info(f"Layer sizes: {connectome['layer_sizes']}")
        
        return connectome
    
    def create_weight_matrices(self, connectome: dict) -> tuple:
        """Generate differentiable weight matrices."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Creating Weight Matrices")
        logger.info("="*60)
        
        config = ConnectivityConfig(
            normalize_by_fanin=True,
            init_strategy='exponential',
            dtype=torch.float32
        )
        
        generator = AdjacencyMatrixGenerator(config)
        
        # Create connectivity masks by layer
        layer_map = {
            ('ORN', 'PN'): (50, 50),
            ('PN', 'KC'): (50, 2000),
            ('KC', 'MBON'): (2000, 50),
            ('MBON', 'DN'): (50, 10),
        }
        
        neurons = connectome['neurons']
        synapses = connectome['synapses']
        
        # Create full weight matrices for simplicity
        # (In production, might create layer-specific matrices)
        
        weights_data = {}
        
        for (src_layer, dst_layer), (n_src, n_dst) in layer_map.items():
            logger.info(f"Creating {src_layer}→{dst_layer} matrix ({n_src}×{n_dst})...")
            
            # Create symbolic weight matrix
            weights = torch.randn(n_src, n_dst, dtype=torch.float32) * 0.1
            mask = torch.zeros(n_src, n_dst, dtype=torch.float32)
            
            # Assign connectivity based on layer
            for synapse in synapses:
                if synapse.pre_type == src_layer and synapse.post_type == dst_layer:
                    # Find indices (simplified - in full implementation would use proper mapping)
                    pre_idx = min(synapse.pre_id % n_src, n_src - 1)
                    post_idx = min(synapse.post_id % n_dst, n_dst - 1)
                    
                    weight_val = np.log1p(synapse.count) * 0.3
                    weights[pre_idx, post_idx] = weight_val
                    mask[pre_idx, post_idx] = 1.0
            
            # Move to device
            weights = weights.to(self.device)
            mask = mask.to(self.device)
            
            weights_data[f'{src_layer}_{dst_layer}'] = (weights, mask)
            
            sparsity = 1.0 - (mask.sum() / mask.numel()).item()
            logger.info(f"  Sparsity: {sparsity:.2%}, Mean weight: {weights[mask>0].mean():.4f}")
        
        return weights_data
    
    def build_circuit(self, connectome: dict, weights_data: dict):
        """Build differentiable olfactory circuit."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: Building Differentiable Circuit")
        logger.info("="*60)
        
        # Prepare weight matrices
        masks = {}
        weights = {}
        
        for key, (w, m) in weights_data.items():
            weights[key] = nn.Parameter(w)
            masks[key] = m
        
        # Create circuit
        self.circuit = DifferentiableOlfactoryCircuit(
            connectivity_data=connectome,
            weights_orn_pn=weights.get('ORN_PN', torch.randn(50, 50, device=self.device).to(self.device) * 0.1),
            weights_pn_kc=weights.get('PN_KC', torch.randn(50, 2000, device=self.device).to(self.device) * 0.01),
            weights_kc_mbon=weights.get('KC_MBON', torch.randn(2000, 50, device=self.device).to(self.device) * 0.01),
            weights_mbon_dn=weights.get('MBON_DN', torch.randn(50, 10, device=self.device).to(self.device) * 0.1),
            masks=masks,
            device=self.device,
            learnable=True
        )
        
        logger.info(f"Circuit created on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.circuit.parameters()):,}")
        logger.info(f"Learnable parameters: {sum(p.numel() for p in self.circuit.parameters() if p.requires_grad):,}")
    
    def setup_training(self):
        """Setup training infrastructure."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: Setting up Training")
        logger.info("="*60)
        
        # Loss function
        loss_fn = CombinedLoss(
            navigation_weight=self.config.get('nav_weight', 1.0),
            energy_weight=self.config.get('energy_weight', 0.1),
            sparsity_weight=self.config.get('sparse_weight', 0.01),
            activity_weight=self.config.get('activity_weight', 0.001),
            goal_position=(50.0, 50.0, 0.0)
        )
        
        # Training config
        train_config = TrainingConfig(
            num_episodes=self.config.get('num_episodes', 100),
            episode_length_steps=self.config.get('episode_length', 5000),
            learning_rate=self.config.get('learning_rate', 0.001),
            optimizer=self.config.get('optimizer', 'adam'),
            use_scheduler=self.config.get('use_scheduler', True),
            device=str(self.device)
        )
        
        # Trainer
        self.trainer = DMNTrainer(
            network=self.circuit,
            loss_function=loss_fn,
            config=train_config
        )
        
        logger.info("Training infrastructure ready")
    
    def train(self):
        """Execute training."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 5: Training Network with BPTT")
        logger.info("="*60)
        
        num_episodes = self.config.get('num_episodes', 100)
        
        episode_losses = self.trainer.train_episode(num_episodes=num_episodes)
        
        self.results['training_losses'] = episode_losses
        
        logger.info(f"Training complete!")
        logger.info(f"Best loss: {self.trainer.best_loss:.4f}")
    
    def evaluate(self):
        """Evaluate trained network."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 6: Evaluating Learned Network")
        logger.info("="*60)
        
        self.circuit.eval()
        with torch.no_grad():
            # Rollout test trajectory
            trajectory = self.trainer.rollout_episode(
                episode_length=5000,
                goal_position=(50.0, 50.0, 0.0)
            )
            
            # Compute metrics
            metrics = TrajectoryAnalyzer.compute_trajectory_metrics(
                trajectory['positions'],
                trajectory['velocities'],
                goal=(50.0, 50.0, 0.0)
            )
            
            # Compute neural statistics
            spike_stats = {}
            for layer, spikes in trajectory['spikes'].items():
                spike_stats[layer] = {
                    'mean_rate': spikes.float().mean().item(),
                    'max_rate': spikes.float().max().item(),
                    'sparsity': (1 - spikes.float().mean()).item()
                }
            
            self.results['final_metrics'] = {
                'trajectory_metrics': metrics,
                'spike_statistics': spike_stats
            }
            
            # Log results
            logger.info("\nFinal Trajectory Metrics:")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.4f}")
            
            logger.info("\nNeural Activity Statistics:")
            for layer, stats in spike_stats.items():
                logger.info(f"  {layer}: mean_rate={stats['mean_rate']:.4f} Hz, "
                           f"sparsity={stats['sparsity']:.2%}")
    
    def save_results(self):
        """Save training results."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 7: Saving Results")
        logger.info("="*60)
        
        # Create results directory
        results_dir = Path('dmn_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_json = results_dir / 'training_results.json'
        
        # Convert torch tensors to lists for JSON serialization
        results_serializable = {
            'config': self.results['config'],
            'training_losses': [
                {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss.items()}
                for loss in self.results['training_losses']
            ],
            'final_metrics': {
                'trajectory_metrics': {k: float(v) for k, v in self.results['final_metrics']['trajectory_metrics'].items()},
                'spike_statistics': self.results['final_metrics']['spike_statistics']
            }
        }
        
        with open(results_json, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {results_json}")
        
        # Save model
        model_path = results_dir / 'trained_circuit.pt'
        self.trainer.save_checkpoint('trained_circuit.pt')
        
        logger.info(f"\nExperiment complete!")
        logger.info(f"Results directory: {results_dir}")
        
        return results_dir
    
    def run(self):
        """Execute complete experiment."""
        try:
            connectome = self.setup_connectome()
            weights_data = self.create_weight_matrices(connectome)
            self.build_circuit(connectome, weights_data)
            self.setup_training()
            self.train()
            self.evaluate()
            results_dir = self.save_results()
            
            return results_dir
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='NeuroMechFly DMN Training Experiment'
    )
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of training episodes')
    parser.add_argument('--episode-length', type=int, default=5000,
                       help='Steps per episode')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', default='adam',
                       help='Optimizer (adam, sgd, rmsprop)')
    parser.add_argument('--nav-weight', type=float, default=1.0,
                       help='Navigation loss weight')
    parser.add_argument('--energy-weight', type=float, default=0.1,
                       help='Energy loss weight')
    parser.add_argument('--use-real-flywire', action='store_true',
                       help='Use real FlyWire data (requires credentials)')
    
    args = parser.parse_args()
    
    config = {
        'num_episodes': args.num_episodes,
        'episode_length': args.episode_length,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'nav_weight': args.nav_weight,
        'energy_weight': args.energy_weight,
        'sparse_weight': 0.01,
        'activity_weight': 0.001,
        'use_real_flywire': args.use_real_flywire,
    }
    
    logger.info("\n" + "="*60)
    logger.info("NeuroMechFly DMN Training")
    logger.info("="*60)
    logger.info(f"Configuration:")
    for key, val in config.items():
        logger.info(f"  {key}: {val}")
    
    # Run experiment
    experiment = DMNExperiment(config)
    results_dir = experiment.run()
    
    print(f"\n✓ Training complete! Results saved to {results_dir}")


if __name__ == '__main__':
    main()
