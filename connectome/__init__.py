"""
FlyWire Connectome Package

Handles integration with FlyWire connectome database and connectivity
data management for differentiable neural network optimization.
"""

from .fetch_data import FlyWireConnectome, SynapseConnection, NeuronMetadata
from .adjacency_matrix import AdjacencyMatrixGenerator, ConnectivityConfig

__all__ = [
    'FlyWireConnectome',
    'SynapseConnection',
    'NeuronMetadata',
    'AdjacencyMatrixGenerator',
    'ConnectivityConfig',
]
