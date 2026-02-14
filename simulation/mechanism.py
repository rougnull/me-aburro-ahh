"""
Differentiable Neural Mechanism Models using PyTorch

Implements LIF spiking neural networks as differentiable RNN cells
compatible with automatic differentiation and backpropagation through time.

Key Features:
- Differentiable LIF neuron dynamics
- Learnable membrane time constants and thresholds
- Sparse connectivity support
- Surrogate gradient descent for spike computation
- PyTorch autograd optimization

Author: NeuroMechFly DMN Framework
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for spike function.
    
    In forward pass: Returns step function (spike = 1 if V > threshold, else 0)
    In backward pass: Uses smooth surrogate to provide gradients
    
    This allows training with discrete spikes while maintaining differentiability.
    """
    
    @staticmethod
    def forward(ctx, x, threshold=0.0):
        """
        Forward pass: spike function.
        
        Args:
            x: Membrane potential
            threshold: Spike threshold
        
        Returns:
            Binary spike: 1 if x > threshold, else 0
        """
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: use soft surrogate for gradients.
        
        Uses fast sigmoid as surrogate: σ(β·x) for smooth approximation
        """
        x, = ctx.saved_tensors
        threshold = ctx.threshold
        beta = 5.0  # Sharpness of surrogate
        
        # Fast sigmoid surrogate
        grad_surr = beta * torch.sigmoid(beta * (x - threshold)) * (1 - torch.sigmoid(beta * (x - threshold)))
        
        return grad_output * grad_surr, None


class LIFCell(nn.Module):
    """
    Single Leaky Integrate-and-Fire neuron as differentiable RNN cell.
    
    Dynamics:
    τ·dV/dt = -V + R·I_syn
    
    Where:
    - V: membrane potential
    - τ: membrane time constant (learnable)
    - R: input resistance
    - I_syn: synaptic input current
    
    Spikes when V crosses threshold θ (learnable).
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_ms: float = 20.0,
        threshold_mv: float = -50.0,
        reset_mv: float = -70.0,
        learnable_tau: bool = False,
        learnable_threshold: bool = False,
        dt_ms: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize LIF neuron population.
        
        Args:
            n_neurons: Number of neurons in this layer
            tau_ms: Membrane time constant (ms)
            threshold_mv: Spike threshold (mV)
            reset_mv: Reset potential after spike (mV)
            learnable_tau: Make τ learnable per neuron
            learnable_threshold: Make θ learnable per neuron
            dt_ms: Integration timestep (ms)
            device: torch device
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.dt_ms = dt_ms
        self.device = device or torch.device('cpu')
        
        # Decay constant: exp(-dt/τ)
        self.base_tau = tau_ms
        self.threshold_mv = threshold_mv
        self.reset_mv = reset_mv
        
        # Learnable parameters
        if learnable_tau:
            self.log_tau = nn.Parameter(
                torch.full((n_neurons,), np.log(tau_ms), device=self.device)
            )
        else:
            self.register_buffer(
                'log_tau',
                torch.full((n_neurons,), np.log(tau_ms), device=self.device)
            )
        
        if learnable_threshold:
            self.threshold = nn.Parameter(
                torch.full((n_neurons,), threshold_mv, device=self.device)
            )
        else:
            self.register_buffer(
                'threshold',
                torch.full((n_neurons,), threshold_mv, device=self.device)
            )
        
        self.reset_potential = torch.tensor(reset_mv, device=self.device)
        
        # State variables (not learnable, tracked through time)
        self.register_buffer('V', torch.full((n_neurons,), reset_mv, device=self.device))
        self.register_buffer('spikes', torch.zeros((n_neurons,), device=self.device))
        
        logger.info(f"LIFCell initialized: n_neurons={n_neurons}, τ={tau_ms}ms, θ={threshold_mv}mV")
    
    @property
    def tau(self) -> torch.Tensor:
        """Get actual time constant from log parameterization."""
        return torch.exp(self.log_tau)
    
    @property
    def decay(self) -> torch.Tensor:
        """Get decay factor exp(-dt/τ) for current timestep."""
        return torch.exp(-self.dt_ms / self.tau)
    
    def forward(
        self,
        I_syn: torch.Tensor,
        V_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single integration step of LIF neuron.
        
        Args:
            I_syn: Synaptic input current (n_neurons,)
            V_prev: Previous membrane potential (default: use stored self.V)
            
        Returns:
            Tuple of:
            - spikes: Binary spike output (n_neurons,)
            - V_new: Updated membrane potential (n_neurons,)
        """
        if V_prev is None:
            V_prev = self.V
        
        # Ensure proper shapes
        I_syn = I_syn.reshape(-1)
        V_prev = V_prev.reshape(-1)
        
        # LIF dynamics: V_new = decay·V_prev + (1-decay)·I_syn
        # This is the discretized form of τ·dV/dt = -V + I_syn
        decay = self.decay
        V_new = decay * V_prev + (1 - decay) * I_syn
        
        # Compute spikes using surrogate gradient
        spikes = SurrogateGradient.apply(V_new, self.threshold)
        
        # Reset membrane potential after spike
        V_new = V_new * (1 - spikes) + self.reset_potential * spikes
        
        # Store for next step
        self.V = V_new.detach()
        self.spikes = spikes.detach()
        
        return spikes, V_new
    
    def reset_state(self, batch_size: int = 1):
        """Reset neuron state for new episode."""
        self.V = torch.full((batch_size, self.n_neurons), self.reset_mv, device=self.device)
        self.spikes = torch.zeros((batch_size, self.n_neurons), device=self.device)


class DifferentiableOlfactoryCircuit(nn.Module):
    """
    Complete olfactory circuit as differentiable neural network.
    
    Architecture:
    ORN (50) → PN (50) → KC (2000) → MBON (50) → DN (10)
    
    All synaptic weights are learnable (subject to connectivity constraints).
    """
    
    def __init__(
        self,
        connectivity_data: Dict,
        weights_orn_pn: torch.Tensor,
        weights_pn_kc: torch.Tensor,
        weights_kc_mbon: torch.Tensor,
        weights_mbon_dn: torch.Tensor,
        masks: Dict,
        device: Optional[torch.device] = None,
        learnable: bool = True
    ):
        """
        Initialize differentiable olfactory circuit.
        
        Args:
            connectivity_data: Connectome data with layer info
            weights_*: Initial weight matrices for each layer
            masks: Connectivity masks to preserve sparse structure
            device: torch device
            learnable: Whether to make weights trainable
        """
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.learnable = learnable
        
        # Layer sizes
        self.n_orn = 50
        self.n_pn = 50
        self.n_kc = 2000
        self.n_mbon = 50
        self.n_dn = 10
        
        # LIF cells for each population
        self.orn_cells = LIFCell(self.n_orn, tau_ms=15, device=self.device, learnable_tau=False)
        self.pn_cells = LIFCell(self.n_pn, tau_ms=15, device=self.device, learnable_tau=True)
        self.kc_cells = LIFCell(self.n_kc, tau_ms=20, device=self.device, learnable_tau=True)
        self.mbon_cells = LIFCell(self.n_mbon, tau_ms=25, device=self.device, learnable_tau=True)
        self.dn_cells = LIFCell(self.n_dn, tau_ms=30, device=self.device, learnable_tau=True)
        
        # Learnable weights (with connectivity constraints)
        self._register_weights(
            'w_orn_pn', weights_orn_pn, masks.get('mask_orn_pn'),
            learnable and 'ORN_to_PN' not in (masks.get('fixed_connectivity') or [])
        )
        self._register_weights(
            'w_pn_kc', weights_pn_kc, masks.get('mask_pn_kc'),
            learnable
        )
        self._register_weights(
            'w_kc_mbon', weights_kc_mbon, masks.get('mask_kc_mbon'),
            learnable
        )
        self._register_weights(
            'w_mbon_dn', weights_mbon_dn, masks.get('mask_mbon_dn'),
            learnable
        )
        
        # Store masks for weight constraint
        self.register_buffer('mask_orn_pn', masks.get('mask_orn_pn'))
        self.register_buffer('mask_pn_kc', masks.get('mask_pn_kc'))
        self.register_buffer('mask_kc_mbon', masks.get('mask_kc_mbon'))
        self.register_buffer('mask_mbon_dn', masks.get('mask_mbon_dn'))
        
        logger.info(f"DifferentiableOlfactoryCircuit initialized: "
                   f"ORN({self.n_orn}) → PN({self.n_pn}) → KC({self.n_kc}) → "
                   f"MBON({self.n_mbon}) → DN({self.n_dn})")
    
    def _register_weights(
        self,
        name: str,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        learnable: bool = True
    ):
        """Register weight matrix, optionally with connectivity mask."""
        if learnable:
            self.register_parameter(name, nn.Parameter(weights))
        else:
            self.register_buffer(name, weights)
    
    def _apply_connectivity_mask(self, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply connectivity mask to weights (zero out non-existent synapses)."""
        return weights * mask
    
    def forward(
        self,
        odor_input: torch.Tensor,
        return_activations: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Single timestep forward pass through circuit.
        
        Args:
            odor_input: Odor stimulus (n_orn,)
            return_activations: Whether to return all layer activations
            
        Returns:
            Tuple of:
            - dn_spikes: Output spikes from descending neurons
            - activations: Dict of all layer outputs (if return_activations=True)
        """
        activations = {}
        
        # ORN layer: sensory input
        orn_input = odor_input  # Direct odor encoding
        orn_spikes, orn_V = self.orn_cells(orn_input)
        activations['orn'] = orn_spikes
        
        # ORN → PN connections
        w_orn_pn_masked = self._apply_connectivity_mask(self.w_orn_pn, self.mask_orn_pn)
        pn_current = torch.matmul(w_orn_pn_masked.T, orn_spikes)
        pn_spikes, pn_V = self.pn_cells(pn_current)
        activations['pn'] = pn_spikes
        
        # PN → KC connections (sparse random)
        w_pn_kc_masked = self._apply_connectivity_mask(self.w_pn_kc, self.mask_pn_kc)
        kc_current = torch.matmul(w_pn_kc_masked.T, pn_spikes)
        kc_spikes, kc_V = self.kc_cells(kc_current)
        activations['kc'] = kc_spikes
        
        # KC → MBON connections (learnable, convergent)
        w_kc_mbon_masked = self._apply_connectivity_mask(self.w_kc_mbon, self.mask_kc_mbon)
        mbon_current = torch.matmul(w_kc_mbon_masked.T, kc_spikes)
        mbon_spikes, mbon_V = self.mbon_cells(mbon_current)
        activations['mbon'] = mbon_spikes
        
        # MBON → DN connections
        w_mbon_dn_masked = self._apply_connectivity_mask(self.w_mbon_dn, self.mask_mbon_dn)
        dn_current = torch.matmul(w_mbon_dn_masked.T, mbon_spikes)
        dn_spikes, dn_V = self.dn_cells(dn_current)
        activations['dn'] = dn_spikes
        
        if return_activations:
            return dn_spikes, activations
        else:
            return dn_spikes, None
    
    def reset_state(self):
        """Reset all neuron states for new episode."""
        self.orn_cells.reset_state()
        self.pn_cells.reset_state()
        self.kc_cells.reset_state()
        self.mbon_cells.reset_state()
        self.dn_cells.reset_state()
    
    def get_learnable_parameters(self) -> List[nn.Parameter]:
        """Get list of all learnable parameters."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append((name, param))
        return params
    
    def get_statistics(self) -> Dict:
        """Compute statistics about network weights."""
        stats = {
            'n_learnable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'w_orn_pn_mean': self.w_orn_pn.mean().item(),
            'w_pn_kc_mean': self.w_pn_kc.mean().item(),
            'w_kc_mbon_mean': self.w_kc_mbon.mean().item(),
            'w_mbon_dn_mean': self.w_mbon_dn.mean().item(),
        }
        return stats


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing Differentiable LIF Neurons")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create simple LIF population
    n_neurons = 100
    lif = LIFCell(n_neurons, tau_ms=20.0, threshold_mv=-50.0, device=device, learnable_tau=True)
    
    # Test with random input
    print(f"\nFiring a 100-neuron LIF population for 100ms...")
    spike_raster = []
    
    for step in range(100):
        # Random input current
        I_in = torch.randn(n_neurons, device=device) * 0.5 + 0.2
        spikes, V = lif(I_in)
        spike_raster.append(spikes.detach().cpu().numpy())
    
    spike_raster = np.array(spike_raster)
    firing_rate = spike_raster.mean() * 1000  # Convert to Hz (assuming 1ms timesteps)
    
    print(f"Mean firing rate: {firing_rate:.1f} Hz")
    print(f"Spike raster shape: {spike_raster.shape}")
    print(f"Sparsity: {1 - spike_raster.mean():.2%}")
    
    # Check that gradients can flow
    print(f"\nTesting backpropagation...")
    lif.reset_state()
    
    total_loss = 0
    for step in range(10):
        I_in = torch.randn(n_neurons, device=device) * 0.5
        spikes, V = lif(I_in)
        loss = spikes.sum()  # Simple loss: maximize spikes
        total_loss = total_loss + loss
    
    total_loss.backward()
    
    # Check that gradients were computed
    has_gradients = any(p.grad is not None for p in lif.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    if lif.log_tau.grad is not None:
        print(f"τ gradient norm: {lif.log_tau.grad.norm().item():.4f}")
