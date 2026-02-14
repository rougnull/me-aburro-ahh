"""
NeuroMechFly DMN Framework Package Initialization

Provides convenient imports for the differentiable mechanical networks
optimization framework.

Usage:
    from neuromechfly import DMNTrainer, FlyWireConnectome, LIFCell
    
    # Or direct imports:
    from connectome.fetch_data import FlyWireConnectome
    from simulation.mechanism import DifferentiableOlfactoryCircuit
    from training.optimizer import DMNTrainer
"""

__version__ = '1.0.0'
__author__ = 'NeuroMechFly DMN Team'
__description__ = 'Differentiable neural circuit optimization for embodied sensorimotor control'

# Core imports
try:
    from connectome.fetch_data import FlyWireConnectome, SynapseConnection, NeuronMetadata
    from connectome.adjacency_matrix import AdjacencyMatrixGenerator, ConnectivityConfig
    
    from simulation.mechanism import (
        LIFCell, 
        DifferentiableOlfactoryCircuit, 
        SurrogateGradient
    )
    
    from training.loss_functions import (
        NavigationLoss,
        EnergyLoss,
        SparsityLoss,
        ActivityRegularizationLoss,
        CombinedLoss,
        TrajectoryAnalyzer
    )
    
    from training.optimizer import DMNTrainer, TrainingConfig
    
    __all__ = [
        # Connectome
        'FlyWireConnectome',
        'SynapseConnection',
        'NeuronMetadata',
        'AdjacencyMatrixGenerator',
        'ConnectivityConfig',
        
        # Neural mechanisms
        'LIFCell',
        'DifferentiableOlfactoryCircuit',
        'SurrogateGradient',
        
        # Loss functions
        'NavigationLoss',
        'EnergyLoss',
        'SparsityLoss',
        'ActivityRegularizationLoss',
        'CombinedLoss',
        'TrajectoryAnalyzer',
        
        # Training
        'DMNTrainer',
        'TrainingConfig',
    ]

except ImportError as e:
    print(f"Warning: Could not import all DMN components: {e}")
    print("Some modules may not be available.")
