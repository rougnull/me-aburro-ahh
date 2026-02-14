"""
Training Package for Differentiable Mechanical Networks

Provides task-based learning infrastructure including loss functions,
optimization loops, and trajectory analysis tools.
"""

from .loss_functions import (
    NavigationLoss,
    EnergyLoss,
    SparsityLoss,
    ActivityRegularizationLoss,
    CombinedLoss,
    TrajectoryAnalyzer
)

from .optimizer import DMNTrainer, TrainingConfig

__all__ = [
    'NavigationLoss',
    'EnergyLoss',
    'SparsityLoss',
    'ActivityRegularizationLoss',
    'CombinedLoss',
    'TrajectoryAnalyzer',
    'DMNTrainer',
    'TrainingConfig',
]
