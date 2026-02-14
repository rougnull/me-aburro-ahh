"""
Adjacency Matrix Generator for PyTorch Neural Networks

Converts FlyWire connectome data into differentiable weight matrices
compatible with PyTorch-based neural optimization.

Features:
- Sparse matrix support (CSR format)
- Connectivity masks to preserve fixed connectivity patterns
- Multiple normalization schemes
- Weight initialization strategies

Author: NeuroMechFly DMN Framework
Version: 1.0
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, dia_matrix, diags
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConnectivityConfig:
    """Configuration for connectivity matrix generation."""
    use_sparse: bool = True
    normalize_by_fanin: bool = True
    normalize_by_fanout: bool = False
    init_strategy: str = 'exponential'  # 'exponential', 'normal', 'uniform'
    min_weight: float = 0.0
    max_weight: float = 1.0
    dtype: torch.dtype = torch.float32


class AdjacencyMatrixGenerator:
    """
    Generates learnable weight matrices from connectome data.
    
    Preserves connectivity constraints (which synapses can exist) while
    allowing weight magnitudes to be learned via backpropagation.
    """
    
    def __init__(self, config: Optional[ConnectivityConfig] = None):
        """
        Initialize adjacency matrix generator.
        
        Args:
            config: Connectivity configuration
        """
        self.config = config or ConnectivityConfig()
    
    def create_weight_matrix(
        self,
        connectivity_data: Dict,
        normalize: bool = True,
        learnable: bool = True
    ) -> Tuple[torch.nn.Parameter, torch.Tensor]:
        """
        Create learnable weight matrix from connectivity data.
        
        Args:
            connectivity_data: Output from FlyWireConnectome.fetch_olfactory_circuit()
            normalize: Whether to normalize weights
            learnable: Whether to make weights a trainable parameter
            
        Returns:
            Tuple of (weight_matrix, connectivity_mask)
            - weight_matrix: Learnable weights (Parameter if learnable=True)
            - connectivity_mask: Binary mask indicating possible connections
        """
        # Extract data
        neurons = connectivity_data['neurons']
        synapses = connectivity_data['synapses']
        
        n_neurons = len(neurons)
        
        # Create connectivity mask (which synapses CAN exist)
        mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        weights = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        
        # Map cell IDs to indices
        cell_id_to_idx = {
            neuron.cell_id: idx
            for idx, neuron in enumerate(neurons.values())
        }
        
        # Fill connectivity mask and initialize weights
        for synapse in synapses:
            pre_idx = cell_id_to_idx.get(synapse.pre_id)
            post_idx = cell_id_to_idx.get(synapse.post_id)
            
            if pre_idx is None or post_idx is None:
                continue
            
            # Mark connectivity as allowed
            mask[pre_idx, post_idx] = 1.0
            
            # Initialize weight based on synapse count and strategy
            if self.config.init_strategy == 'exponential':
                # Larger weight for more synapses
                weights[pre_idx, post_idx] = min(
                    self.config.max_weight,
                    self.config.min_weight + np.log1p(synapse.count) * 0.3
                )
            elif self.config.init_strategy == 'normal':
                weights[pre_idx, post_idx] = np.random.normal(0.5, 0.1)
                weights[pre_idx, post_idx] = np.clip(
                    weights[pre_idx, post_idx],
                    self.config.min_weight,
                    self.config.max_weight
                )
            else:  # uniform
                weights[pre_idx, post_idx] = np.random.uniform(
                    self.config.min_weight,
                    self.config.max_weight
                )
        
        # Normalize weights
        if normalize:
            weights = self._normalize_weights(weights, mask)
        
        # Convert to torch tensors
        weight_tensor = torch.from_numpy(weights).to(dtype=self.config.dtype)
        mask_tensor = torch.from_numpy(mask).to(dtype=self.config.dtype)
        
        # Make learnable parameter
        if learnable:
            weight_param = nn.Parameter(weight_tensor)
        else:
            weight_param = weight_tensor
        
        logger.info(f"Created weight matrix: shape={weight_tensor.shape}, "
                   f"sparsity={(1-mask.sum()/mask.size):.2%}, "
                   f"learnable={learnable}")
        
        return weight_param, mask_tensor
    
    def _normalize_weights(
        self,
        weights: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Normalize weights according to configuration.
        
        Args:
            weights: Weight matrix
            mask: Connectivity mask
            
        Returns:
            Normalized weight matrix
        """
        weights = weights.copy()
        
        if self.config.normalize_by_fanin:
            # Divide each neuron's inputs by fan-in
            col_sums = np.sum(mask, axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            weights = weights / col_sums
        
        elif self.config.normalize_by_fanout:
            # Divide each neuron's outputs by fan-out
            row_sums = np.sum(mask, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            weights = weights / row_sums
        
        return weights
    
    def create_sparse_weight_matrix(
        self,
        connectivity_data: Dict,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.sparse.FloatTensor, torch.Tensor]:
        """
        Create sparse weight matrix using COO (coordinate) format.
        
        Better for very large, sparse connectomes. PyTorch sparse tensors
        support backprop through most operations.
        
        Args:
            connectivity_data: Output from FlyWireConnectome.fetch_olfactory_circuit()
            device: Device to place tensor on (CPU/GPU)
            
        Returns:
            Tuple of (sparse_weight_matrix, dense_mask)
        """
        neurons = connectivity_data['neurons']
        synapses = connectivity_data['synapses']
        n_neurons = len(neurons)
        
        if device is None:
            device = torch.device('cpu')
        
        # Collect COO format data
        rows = []
        cols = []
        values = []
        
        cell_id_to_idx = {
            neuron.cell_id: idx
            for idx, neuron in enumerate(neurons.values())
        }
        
        for synapse in synapses:
            pre_idx = cell_id_to_idx.get(synapse.pre_id)
            post_idx = cell_id_to_idx.get(synapse.post_id)
            
            if pre_idx is None or post_idx is None:
                continue
            
            rows.append(pre_idx)
            cols.append(post_idx)
            
            # Weight based on synapse count
            w = min(self.config.max_weight,
                   self.config.min_weight + np.log1p(synapse.count) * 0.3)
            values.append(w)
        
        # Create indices and values tensors
        indices = torch.LongTensor([rows, cols]).to(device)
        values = torch.FloatTensor(values).to(device)
        
        # Create sparse tensor
        sparse_weights = torch.sparse.FloatTensor(
            indices,
            values,
            torch.Size([n_neurons, n_neurons])
        ).to(device)
        
        # Create dense mask
        mask = torch.zeros((n_neurons, n_neurons), device=device)
        mask[rows, cols] = 1.0
        
        logger.info(f"Created sparse weight matrix: shape={sparse_weights.shape}, "
                   f"nnz={len(values)}, device={device}")
        
        return sparse_weights, mask
    
    def get_layer_connectivity(
        self,
        connectivity_data: Dict,
        src_layer: str,
        dst_layer: str,
        neurons: Dict
    ) -> np.ndarray:
        """
        Extract connectivity between two specific layers.
        
        Args:
            connectivity_data: Connectome data
            src_layer: Source layer name (e.g., 'PN')
            dst_layer: Destination layer name (e.g., 'KC')
            neurons: Neuron metadata dictionary
            
        Returns:
            Connectivity matrix between the two layers
        """
        synapses = connectivity_data['synapses']
        
        # Get neuron indices for each layer
        src_neuron_indices = [
            idx for idx, neuron in enumerate(neurons.values())
            if neuron.cell_type == src_layer
        ]
        src_neuron_ids = [
            list(neurons.values())[idx].cell_id
            for idx in src_neuron_indices
        ]
        
        dst_neuron_indices = [
            idx for idx, neuron in enumerate(neurons.values())
            if neuron.cell_type == dst_layer
        ]
        dst_neuron_ids = [
            list(neurons.values())[idx].cell_id
            for idx in dst_neuron_indices
        ]
        
        # Create mapping
        src_id_to_layer_idx = {nid: idx for idx, nid in enumerate(src_neuron_ids)}
        dst_id_to_layer_idx = {nid: idx for idx, nid in enumerate(dst_neuron_ids)}
        
        # Create layer connectivity matrix
        layer_conn = np.zeros((len(src_neuron_ids), len(dst_neuron_ids)))
        
        for synapse in synapses:
            if (synapse.pre_id in src_id_to_layer_idx and
                synapse.post_id in dst_id_to_layer_idx):
                
                src_idx = src_id_to_layer_idx[synapse.pre_id]
                dst_idx = dst_id_to_layer_idx[synapse.post_id]
                layer_conn[src_idx, dst_idx] = synapse.count
        
        return layer_conn
    
    def get_statistics(
        self,
        weights: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict:
        """
        Compute statistics about weight matrix.
        
        Args:
            weights: Weight matrix tensor
            mask: Connectivity mask
            
        Returns:
            Dictionary of statistics
        """
        weights_np = weights.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        # Masked weights
        masked_weights = weights_np[mask_np > 0]
        
        stats = {
            'total_neurons': weights_np.shape[0],
            'total_synapses': int(np.sum(mask_np)),
            'sparsity': float(1 - np.sum(mask_np) / mask_np.size),
            'weight_mean': float(np.mean(masked_weights)) if len(masked_weights) > 0 else 0.0,
            'weight_std': float(np.std(masked_weights)) if len(masked_weights) > 0 else 0.0,
            'weight_min': float(np.min(masked_weights)) if len(masked_weights) > 0 else 0.0,
            'weight_max': float(np.max(masked_weights)) if len(masked_weights) > 0 else 0.0,
            'fan_in_mean': float(np.mean(np.sum(mask_np, axis=0))),
            'fan_out_mean': float(np.mean(np.sum(mask_np, axis=1))),
        }
        
        return stats


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from connectome.fetch_data import FlyWireConnectome
    
    print("\n" + "="*60)
    print("Creating Differentiable Weight Matrices")
    print("="*60)
    
    # Get connectivity data
    fetcher = FlyWireConnectome()
    connectome = fetcher.fetch_olfactory_circuit(use_synthetic=True)
    
    # Create weight matrices
    generator = AdjacencyMatrixGenerator()
    weights, mask = generator.create_weight_matrix(connectome, learnable=True)
    
    print(f"\nWeight matrix shape: {weights.shape}")
    print(f"Connectivity mask shape: {mask.shape}")
    print(f"Weights is parameter: {isinstance(weights, nn.Parameter)}")
    
    # Get statistics
    stats = generator.get_statistics(weights, mask)
    print("\nWeight matrix statistics:")
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    # Try sparse format
    print("\n" + "-"*60)
    print("Creating sparse weight matrix...")
    sparse_weights, sparse_mask = generator.create_sparse_weight_matrix(connectome)
    print(f"Sparse weights type: {type(sparse_weights)}")
    print(f"Sparse weights shape: {sparse_weights.shape}")
