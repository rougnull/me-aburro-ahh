"""
Task-based Loss Functions for DMN Training

Defines optimization objectives for learning navigation and behavioral control.

Loss Components:
1. Navigation Loss: Distance to goal/food source
2. Energy Loss: Penalizes excessive movement
3. Sparsity Loss: Maintains KC sparse coding (~2% activity)
4. Activity Regularization: Prevents epileptic firing patterns

Author: NeuroMechFly DMN Framework
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NavigationLoss(nn.Module):
    """
    Loss for odor navigation task: minimize distance to food source.
    
    Reward structure:
    L_nav = ||position - goal||_2 + penalty_if_no_progress
    """
    
    def __init__(
        self,
        goal_position: Tuple[float, float, float] = (50.0, 50.0, 0.0),
        penalty_no_progress: float = 0.1
    ):
        """
        Initialize navigation loss.
        
        Args:
            goal_position: Coordinates of food source in arena
            penalty_no_progress: Penalty if fly doesn't move closer to goal
        """
        super().__init__()
        
        self.goal = torch.tensor(goal_position, dtype=torch.float32)
        self.penalty_no_progress = penalty_no_progress
    
    def forward(
        self,
        positions: torch.Tensor,
        prev_distances: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute navigation loss over trajectory.
        
        Args:
            positions: Position sequence (T, 3) where T is time steps
            prev_distances: Previous distances for progress penalty
            
        Returns:
            Dict with loss components
        """
        # Ensure goal is on same device
        goal = self.goal.to(positions.device)
        
        # Compute distances from each position to goal
        distances = torch.norm(positions - goal.unsqueeze(0), dim=-1)
        
        # Primary loss: minimize distance
        distance_loss = distances.mean()
        
        # Bonus for reaching goal (within 5mm)
        reached_goal = (distances[-1] < 5.0).float()
        goal_bonus = -reached_goal * 10.0  # Large negative loss (reward)
        
        # Progress penalty: penalize if moving away from goal
        if prev_distances is not None:
            progress = prev_distances[0] - distances[-1]  # Positive = moving closer
            progress_penalty = -torch.clamp(progress, min=0) * self.penalty_no_progress
        else:
            progress_penalty = 0.0
        
        total_loss = distance_loss + goal_bonus + progress_penalty
        
        return {
            'distance_loss': distance_loss,
            'goal_bonus': goal_bonus,
            'progress_penalty': progress_penalty,
            'total_loss': total_loss,
            'final_distance': distances[-1]
        }
    
    def set_goal(self, goal_position: Tuple[float, float, float]):
        """Change goal position for training."""
        self.goal = torch.tensor(goal_position, dtype=torch.float32)


class EnergyLoss(nn.Module):
    """
    Energy efficiency loss: penalize excessive velocity.
    
    Biological organisms minimize energy expenditure. High velocity → high energy cost.
    L_energy = ∫ ||velocity||_2^2 dt
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize energy loss.
        
        Args:
            weight: Coefficient for energy penalty
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, velocities: torch.Tensor) -> torch.Tensor:
        """
        Compute energy loss from velocity trajectory.
        
        Args:
            velocities: Velocity sequence (T, 3)
            
        Returns:
            Energy loss
        """
        # Energy is proportional to velocity squared
        speed_squared = torch.sum(velocities ** 2, dim=-1)
        energy_loss = self.weight * speed_squared.mean()
        
        return energy_loss


class SparsityLoss(nn.Module):
    """
    Maintain sparse Kenyon cell representation.
    
    KCs should be sparsely active (~2% of neurons fire in any given timestep).
    This prevents information overload and maintains distinctiveness.
    
    Uses target sparsity loss: ||actual_sparsity - target_sparsity||_2
    """
    
    def __init__(
        self,
        target_sparsity: float = 0.02,
        weight: float = 0.01
    ):
        """
        Initialize sparsity loss.
        
        Args:
            target_sparsity: Desired fraction of KCs firing (0.02 = 2%)
            weight: Coefficient for sparsity penalty
        """
        super().__init__()
        self.target_sparsity = target_sparsity
        self.weight = weight
    
    def forward(self, kc_spikes: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss for KC population.
        
        Args:
            kc_spikes: KC spike raster (T, n_kc)
            
        Returns:
            Sparsity loss
        """
        # Compute average firing rate
        actual_sparsity = kc_spikes.mean()
        
        # L2 distance from target
        sparsity_loss = self.weight * (actual_sparsity - self.target_sparsity) ** 2
        
        return sparsity_loss


class ActivityRegularizationLoss(nn.Module):
    """
    Prevent pathological neural activity (excessive firing).
    
    Regularization terms:
    - L1 regularization on spike counts
    - Prevent unstable feedback loops
    """
    
    def __init__(
        self,
        weight_l1: float = 0.001,
        max_firing_rate_hz: float = 100.0
    ):
        """
        Initialize activity regularization.
        
        Args:
            weight_l1: Coefficient for L1 sparsity penalty
            max_firing_rate_hz: Maximum allowed firing rate
        """
        super().__init__()
        self.weight_l1 = weight_l1
        self.max_firing_rate_hz = max_firing_rate_hz
    
    def forward(
        self,
        spikes: Dict[str, torch.Tensor],
        dt_ms: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute activity regularization penalties.
        
        Args:
            spikes: Dict of spike rasters by layer {'orn': ..., 'pn': ..., etc}
            dt_ms: Integration timestep in milliseconds
            
        Returns:
            Dict of regularization loss components
        """
        losses = {}
        
        # L1 regularization on each layer
        for layer_name, spike_raster in spikes.items():
            l1_loss = self.weight_l1 * torch.mean(torch.abs(spike_raster))
            losses[f'{layer_name}_l1'] = l1_loss
        
        # Check for excessive firing rates
        dt_s = dt_ms / 1000.0
        max_rate_per_step = self.max_firing_rate_hz * dt_s
        
        violation_loss = 0.0
        for layer_name, spike_raster in spikes.items():
            mean_rate = spike_raster.mean() / dt_s
            if mean_rate > self.max_firing_rate_hz:
                violation = (mean_rate - self.max_firing_rate_hz) / self.max_firing_rate_hz
                violation_loss = violation_loss + violation ** 2
        
        losses['firing_rate_violation'] = violation_loss
        losses['total_regularization'] = sum(losses.values())
        
        return losses


class CombinedLoss(nn.Module):
    """
    Composite loss function combining all objectives.
    
    Total Loss = w_nav·L_nav + w_energy·L_energy + w_sparse·L_sparse + w_reg·L_reg
    """
    
    def __init__(
        self,
        navigation_weight: float = 1.0,
        energy_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        activity_weight: float = 0.001,
        goal_position: Tuple[float, float, float] = (50.0, 50.0, 0.0)
    ):
        """
        Initialize combined loss.
        
        Args:
            navigation_weight: Weight for navigation loss
            energy_weight: Weight for energy efficiency loss
            sparsity_weight: Weight for KC sparsity loss
            activity_weight: Weight for activity regularization
            goal_position: Target goal position
        """
        super().__init__()
        
        self.nav_weight = navigation_weight
        self.energy_weight = energy_weight
        self.sparse_weight = sparsity_weight
        self.activity_weight = activity_weight
        
        self.nav_loss = NavigationLoss(goal_position=goal_position)
        self.energy_loss = EnergyLoss(weight=energy_weight)
        self.sparse_loss = SparsityLoss(weight=sparsity_weight)
        self.activity_loss = ActivityRegularizationLoss(weight_l1=activity_weight)
        
        logger.info(f"CombinedLoss initialized: "
                   f"nav={navigation_weight}, energy={energy_weight}, "
                   f"sparse={sparsity_weight}, activity={activity_weight}")
    
    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        spikes: Dict[str, torch.Tensor],
        dt_ms: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss over trajectory.
        
        Args:
            positions: Position sequence (T, 3)
            velocities: Velocity sequence (T, 3)
            spikes: Dict of spike rasters by layer
            dt_ms: Timestep in milliseconds
            
        Returns:
            Dict with all loss components
        """
        losses = {}
        
        # Navigation loss
        nav_results = self.nav_loss(positions)
        losses['nav_distance'] = nav_results['distance_loss']
        losses['nav_goal_bonus'] = nav_results['goal_bonus']
        
        # Energy loss
        energy = self.energy_loss(velocities)
        losses['energy'] = energy
        
        # Sparsity loss (KC only)
        if 'kc' in spikes:
            sparse = self.sparse_loss(spikes['kc'])
            losses['sparsity'] = sparse
        
        # Activity regularization
        activity_results = self.activity_loss(spikes, dt_ms=dt_ms)
        for key, val in activity_results.items():
            losses[f'activity_{key}'] = val
        
        # Total weighted loss
        total = (
            self.nav_weight * losses['nav_distance'] +
            losses['nav_goal_bonus'] +
            self.energy_weight * losses['energy'] +
            (self.sparse_weight * losses['sparsity'] if 'sparsity' in losses else 0.0) +
            self.activity_weight * activity_results['total_regularization']
        )
        
        losses['total'] = total
        
        return losses
    
    def set_weights(
        self,
        nav: float = None,
        energy: float = None,
        sparse: float = None,
        activity: float = None
    ):
        """Adjust loss weights during training."""
        if nav is not None:
            self.nav_weight = nav
            self.nav_loss.weight = nav
        if energy is not None:
            self.energy_weight = energy
            self.energy_loss.weight = energy
        if sparse is not None:
            self.sparse_weight = sparse
            self.sparse_loss.weight = sparse
        if activity is not None:
            self.activity_weight = activity
            self.activity_loss.weight_l1 = activity


class TrajectoryAnalyzer:
    """
    Analyze optimized trajectories to extract behavioral insights.
    """
    
    @staticmethod
    def compute_trajectory_metrics(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        goal: Tuple[float, float, float]
    ) -> Dict[str, float]:
        """
        Compute comprehensive trajectory metrics.
        
        Args:
            positions: Position sequence (T, 3)
            velocities: Velocity sequence (T, 3)
            goal: Goal position
            
        Returns:
            Dictionary of metrics
        """
        goal_tensor = torch.tensor(goal)
        
        # Distance metrics
        distances = torch.norm(positions - goal_tensor, dim=-1)
        
        # Speed metrics
        speeds = torch.norm(velocities, dim=-1)
        
        metrics = {
            'total_distance_traveled': torch.norm(
                torch.diff(positions, dim=0), dim=-1
            ).sum().item(),
            'final_distance_to_goal': distances[-1].item(),
            'min_distance_to_goal': distances.min().item(),
            'mean_distance_to_goal': distances.mean().item(),
            'mean_speed': speeds.mean().item(),
            'max_speed': speeds.max().item(),
            'exploration_area': TrajectoryAnalyzer._estimate_exploration_area(positions).item(),
        }
        
        return metrics
    
    @staticmethod
    def _estimate_exploration_area(positions: torch.Tensor) -> torch.Tensor:
        """Estimate 2D area explored by projection onto XY plane."""
        xy_positions = positions[:, :2]
        x_range = (xy_positions[:, 0].max() - xy_positions[:, 0].min()).item()
        y_range = (xy_positions[:, 1].max() - xy_positions[:, 1].min()).item()
        return torch.tensor(x_range * y_range)


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy trajectory
    T = 100  # 100 timesteps
    positions = torch.randn(T, 3, device=device) * 10 + 50  # Around goal at (50, 50, 0)
    positions[:, 2] = 0  # Keep Z=0 (2D motion)
    velocities = torch.randn(T, 3, device=device) * 0.5
    spikes = {
        'orn': torch.rand(T, 50, device=device) > 0.95,
        'pn': torch.rand(T, 50, device=device) > 0.95,
        'kc': torch.rand(T, 2000, device=device) > 0.98,
        'mbon': torch.rand(T, 50, device=device) > 0.95,
        'dn': torch.rand(T, 10, device=device) > 0.90,
    }
    
    # Test individual losses
    print("\n1. Navigation Loss")
    nav_loss = NavigationLoss(goal_position=(55, 50, 0))
    nav_result = nav_loss(positions)
    print(f"   Distance loss: {nav_result['distance_loss']:.4f}")
    print(f"   Final distance: {nav_result['final_distance']:.4f}")
    
    print("\n2. Energy Loss")
    energy_loss = EnergyLoss(weight=0.1)
    energy = energy_loss(velocities)
    print(f"   Energy loss: {energy:.4f}")
    
    print("\n3. Sparsity Loss")
    sparse_loss = SparsityLoss(target_sparsity=0.02)
    sparse = sparse_loss(spikes['kc'])
    print(f"   Sparsity loss: {sparse:.4f}")
    print(f"   Actual KC sparsity: {1 - spikes['kc'].float().mean():.2%}")
    
    print("\n4. Combined Loss")
    combined = CombinedLoss(
        navigation_weight=1.0,
        energy_weight=0.1,
        sparsity_weight=0.01,
        activity_weight=0.001
    )
    combined_result = combined(positions, velocities, spikes)
    print(f"   Total loss: {combined_result['total']:.4f}")
    for key, val in combined_result.items():
        if isinstance(val, torch.Tensor):
            print(f"   {key}: {val:.4f}")
    
    print("\n5. Trajectory Metrics")
    metrics = TrajectoryAnalyzer.compute_trajectory_metrics(positions, velocities, (55, 50, 0))
    for key, val in metrics.items():
        print(f"   {key}: {val:.4f}")
