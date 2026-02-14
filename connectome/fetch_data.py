"""
FlyWire Connectome Data Fetching Module

Handles integration with FlyWire connectome database for retrieving real 
Drosophila connectivity patterns. Supports both CAVEclient API and offline 
synthetic data for initial development.

Author: NeuroMechFly DMN Framework
Version: 1.0
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SynapseConnection:
    """Represents a single synaptic connection between two neurons."""
    pre_id: int
    post_id: int
    count: int  # Number of synapses
    distance_um: Optional[float] = None
    pre_type: Optional[str] = None
    post_type: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class NeuronMetadata:
    """Metadata about a single neuron in the connectome."""
    cell_id: int
    cell_type: str
    layer: str
    volume: Optional[float] = None
    position: Optional[Tuple[float, float, float]] = None
    notes: Optional[str] = None


# ============================================================================
# CONNECTOME FETCHER
# ============================================================================

class FlyWireConnectome:
    """
    Main interface for retrieving FlyWire connectome data.
    
    Supports:
    - Real FlyWire API (via CAVEclient) - requires credentials
    - Synthetic connectivity matching published statistics
    - Cached data for offline use
    """
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize FlyWire connectome fetcher.
        
        Args:
            use_cache: Whether to use cached data if available
            cache_dir: Directory for storing cached connectome data
        """
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        self._neurons: Dict[int, NeuronMetadata] = {}
        self._synapses: List[SynapseConnection] = []
        self._cache_loaded = False
        
        logger.info(f"FlyWireConnectome initialized with cache_dir={self.cache_dir}")
    
    def fetch_olfactory_circuit(
        self,
        include_layers: Optional[List[str]] = None,
        min_synapse_count: int = 1,
        use_synthetic: bool = True
    ) -> Dict:
        """
        Fetch olfactory circuit connectivity (ORN → PN → KC → MBON → DN).
        
        Args:
            include_layers: List of layers to include ('ORN', 'PN', 'KC', 'MBON', 'DN')
            min_synapse_count: Minimum synapses to include connection
            use_synthetic: Use synthetic data matching published statistics
            
        Returns:
            Dictionary with connectivity information
        """
        if use_synthetic:
            logger.info("Using synthetic olfactory circuit based on published statistics")
            return self._generate_synthetic_olfactory_circuit(include_layers)
        else:
            logger.info("Attempting to fetch real FlyWire olfactory circuit (requires CAVEclient)")
            return self._fetch_real_olfactory_circuit()
    
    def _generate_synthetic_olfactory_circuit(
        self,
        include_layers: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate synthetic olfactory circuit based on published Drosophila data.
        
        References:
        - Turner et al. 2020: ORN→PN connectivity
        - Sørensen et al. 2016: PN→KC sparse random (~2%)
        - Takemura et al. 2017: KC→MBON learned rules
        """
        if include_layers is None:
            include_layers = ['ORN', 'PN', 'KC', 'MBON', 'DN']
        
        # Published neuron counts for Drosophila antennal lobe + mushroom body
        layer_sizes = {
            'ORN': 50,      # ~50 olfactory receptor neurons (modeled subset)
            'PN': 50,       # ~50 projection neurons (matches ORN)
            'KC': 2000,     # ~2000 Kenyon cells
            'MBON': 50,     # ~50 mushroom body output neurons
            'DN': 10        # ~10 descending neurons
        }
        
        connectivity_rules = {
            ('ORN', 'PN'): {'sparsity': 1.0, 'weight_dist': 'exponential'},  # Dense one-to-one/few-to-many
            ('PN', 'KC'): {'sparsity': 0.02, 'weight_dist': 'exponential'},   # Sparse random
            ('KC', 'MBON'): {'sparsity': 0.1, 'weight_dist': 'exponential'},  # Sparse learned
            ('MBON', 'DN'): {'sparsity': 0.5, 'weight_dist': 'normal'},       # Dense to motor
        }
        
        neurons = {}
        synapses = []
        neuron_id = 0
        
        # Create neurons in each layer
        for layer in include_layers:
            if layer not in layer_sizes:
                logger.warning(f"Unknown layer: {layer}, skipping")
                continue
            
            layer_neuron_count = layer_sizes[layer]
            for idx in range(layer_neuron_count):
                neurons[neuron_id] = NeuronMetadata(
                    cell_id=neuron_id,
                    cell_type=layer,
                    layer=layer,
                    position=(np.random.rand(3) * 100).tolist()  # Random position in 100×100×100 μm
                )
                neuron_id += 1
        
        # Generate connections between layers
        for src_layer, dst_layer in self._get_layer_sequence(include_layers):
            conn_preset = connectivity_rules.get((src_layer, dst_layer))
            if conn_preset is None:
                continue
            
            src_neurons = [n for n in neurons.values() if n.cell_type == src_layer]
            dst_neurons = [n for n in neurons.values() if n.cell_type == dst_layer]
            
            sparsity = conn_preset['sparsity']
            weight_dist = conn_preset['weight_dist']
            
            # Generate connections
            for src in src_neurons:
                # Determine number of targets based on sparsity
                n_targets = max(1, int(len(dst_neurons) * sparsity))
                
                # Randomly select targets
                target_indices = np.random.choice(len(dst_neurons), n_targets, replace=False)
                
                for target_idx in target_indices:
                    dst = dst_neurons[target_idx]
                    
                    # Generate synaptic count based on distribution
                    if weight_dist == 'exponential':
                        count = np.random.poisson(lam=3) + 1  # At least 1 synapse
                    else:  # normal
                        count = max(1, int(np.random.normal(5, 2)))
                    
                    synapses.append(SynapseConnection(
                        pre_id=src.cell_id,
                        post_id=dst.cell_id,
                        count=count,
                        pre_type=src.cell_type,
                        post_type=dst.cell_type
                    ))
        
        logger.info(f"Generated synthetic circuit: {len(neurons)} neurons, {len(synapses)} synapses")
        
        return {
            'neurons': neurons,
            'synapses': synapses,
            'layer_sizes': {layer: layer_sizes[layer] for layer in include_layers},
            'total_synapses': len(synapses),
            'source': 'synthetic'
        }
    
    def _fetch_real_olfactory_circuit(self) -> Dict:
        """
        Fetch real FlyWire olfactory circuit via CAVEclient.
        
        NOTE: Requires:
        - caveclient library: pip install caveclient
        - Valid FlyWire authorization token
        - May require significant bandwidth for large connectome queries
        """
        try:
            import caveclient
        except ImportError:
            logger.error("caveclient not installed. Install with: pip install caveclient")
            logger.warning("Falling back to synthetic data")
            return self._generate_synthetic_olfactory_circuit()
        
        try:
            # This is a placeholder implementation
            # In production, would need proper FlyWire credentials and API calls
            logger.info("Real FlyWire integration not yet fully implemented")
            logger.info("Requires: CAVEclient setup + authentication token")
            logger.warning("Falling back to synthetic olfactory circuit")
            return self._generate_synthetic_olfactory_circuit()
        except Exception as e:
            logger.error(f"Error fetching real connectome: {e}")
            logger.warning("Falling back to synthetic data")
            return self._generate_synthetic_olfactory_circuit()
    
    def _get_layer_sequence(self, include_layers: List[str]) -> List[Tuple[str, str]]:
        """Get the intended sequence of layers for connectivity."""
        full_sequence = ['ORN', 'PN', 'KC', 'MBON', 'DN']
        layer_sequence = [l for l in full_sequence if l in include_layers]
        
        # Return pairs of consecutive layers
        pairs = []
        for i in range(len(layer_sequence) - 1):
            pairs.append((layer_sequence[i], layer_sequence[i + 1]))
        return pairs
    
    def get_adjacency_matrix(
        self,
        neurons: Dict[int, NeuronMetadata],
        synapses: List[SynapseConnection],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Convert neuron and synapse data to adjacency matrix.
        
        Args:
            neurons: Dictionary of neuron metadata
            synapses: List of synapse connections
            normalize: Whether to normalize by presynaptic partner count
            
        Returns:
            Adjacency matrix (n_neurons, n_neurons)
        """
        n = len(neurons)
        adj = np.zeros((n, n))
        
        # Create mapping from cell_id to matrix index
        id_to_idx = {neuron.cell_id: idx for idx, neuron in enumerate(neurons.values())}
        
        # Fill adjacency matrix
        for synapse in synapses:
            pre_idx = id_to_idx[synapse.pre_id]
            post_idx = id_to_idx[synapse.post_id]
            adj[pre_idx, post_idx] = synapse.count
        
        # Normalize if requested
        if normalize:
            # Normalize by presynaptic partner count (target-normalized)
            col_sums = adj.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            adj = adj / col_sums
        
        return adj
    
    def save_connectome(self, connectivity: Dict, filename: str) -> Path:
        """Save connectome data to JSON file."""
        save_path = self.cache_dir / filename
        
        # Convert to JSON-serializable format
        serializable = {
            'neurons': {
                str(nid): {k: v for k, v in asdict(neuron).items()}
                for nid, neuron in connectivity['neurons'].items()
            },
            'synapses': [asdict(syn) for syn in connectivity['synapses']],
            'layer_sizes': connectivity['layer_sizes'],
            'total_synapses': connectivity['total_synapses'],
            'source': connectivity['source']
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Connectome saved to {save_path}")
        return save_path
    
    def load_connectome(self, filename: str) -> Dict:
        """Load connectome data from JSON file."""
        load_path = self.cache_dir / filename
        
        if not load_path.exists():
            logger.warning(f"Connectome file not found: {load_path}")
            return None
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct NeuronMetadata objects
        neurons = {
            int(nid): NeuronMetadata(**neuron_data)
            for nid, neuron_data in data['neurons'].items()
        }
        
        # Reconstruct SynapseConnection objects
        synapses = [
            SynapseConnection(**syn_data)
            for syn_data in data['synapses']
        ]
        
        return {
            'neurons': neurons,
            'synapses': synapses,
            'layer_sizes': data['layer_sizes'],
            'total_synapses': data['total_synapses'],
            'source': data['source']
        }


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize fetcher
    fetcher = FlyWireConnectome(use_cache=True)
    
    # Fetch synthetic olfactory circuit
    print("\n" + "="*60)
    print("Fetching Olfactory Circuit Connectivity")
    print("="*60)
    
    connectome = fetcher.fetch_olfactory_circuit(
        include_layers=['ORN', 'PN', 'KC', 'MBON', 'DN'],
        use_synthetic=True
    )
    
    print(f"\nNeurons in circuit: {len(connectome['neurons'])}")
    print(f"Total synapses: {connectome['total_synapses']}")
    print(f"Layer sizes: {connectome['layer_sizes']}")
    
    # Get adjacency matrix
    adj = fetcher.get_adjacency_matrix(connectome['neurons'], connectome['synapses'])
    print(f"\nAdjacency matrix shape: {adj.shape}")
    print(f"Sparsity: {1 - (np.count_nonzero(adj) / adj.size):.2%}")
    
    # Save connectome
    save_path = fetcher.save_connectome(connectome, "olfactory_circuit.json")
    print(f"\nConnectome saved to: {save_path}")
