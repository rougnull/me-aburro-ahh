"""
Olfactory circuit implementation using a simple neural network model.
Models the Drosophila olfactory pathway:
ORN (sensors) -> PN (glomeruli) -> KC (mushroom body) -> MBON (output)
"""

import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class OlfactoryCircuit:
    """
    Simple neural circuit model for olfactory processing.
    Implements a basic feed-forward network with LIF (Leaky Integrate-and-Fire) neurons.
    
    Circuit structure:
    - ORN: Olfactory Receptor Neurons (sensory input layer)
    - PN: Projection Neurons (primary processing)
    - KC: Kenyon Cells (mushroom body, associative learning)
    - MBON: Mushroom Body Output Neurons (valence encoding)
    - DN: Descending Neurons (motor output)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the olfactory circuit.
        
        Args:
            config: Configuration dictionary with neuron counts and parameters
        """
        self.config = config
        
        # Layer sizes
        self.n_orn = config.get('neurons', {}).get('orn_count', 50)
        self.n_pn = config.get('neurons', {}).get('pn_count', 20)
        self.n_kc = config.get('neurons', {}).get('kc_count', 2000)
        self.n_mbon = config.get('neurons', {}).get('mbon_count', 34)
        self.n_dn = config.get('neurons', {}).get('dn_count', 10)
        
        # Neuron states
        self.orn_membrane = np.random.uniform(-70, -60, self.n_orn)
        self.pn_membrane = np.random.uniform(-70, -60, self.n_pn)
        self.kc_membrane = np.random.uniform(-70, -60, self.n_kc)
        self.mbon_membrane = np.random.uniform(-70, -60, self.n_mbon)
        self.dn_membrane = np.random.uniform(-70, -60, self.n_dn)
        
        # Spikes (binary)
        self.orn_spikes = np.zeros(self.n_orn, dtype=bool)
        self.pn_spikes = np.zeros(self.n_pn, dtype=bool)
        self.kc_spikes = np.zeros(self.n_kc, dtype=bool)
        self.mbon_spikes = np.zeros(self.n_mbon, dtype=bool)
        self.dn_spikes = np.zeros(self.n_dn, dtype=bool)
        
        # Synaptic weights
        self.w_orn_pn = np.random.normal(0.5, 0.1, (self.n_pn, self.n_orn))
        self.w_pn_kc = np.random.normal(0.1, 0.02, (self.n_kc, self.n_pn))
        self.w_kc_mbon = np.random.normal(0.2, 0.05, (self.n_mbon, self.n_kc))
        self.w_mbon_dn = np.random.normal(0.3, 0.1, (self.n_dn, self.n_mbon))
        
        # Parameters
        self.tau_m = config.get('temporal', {}).get('membrane_time_constant', 0.01)
        self.spike_threshold = config.get('temporal', {}).get('spike_threshold', -50.0)
        self.rest_potential = config.get('temporal', {}).get('resting_potential', -70.0)
        
        self.lateral_inhibition = config.get('synapses', {}).get('lateral_inhibition_strength', 0.05)
        self.spontaneous_activity = config.get('synapses', {}).get('spontaneous_activity', 0.01)
        
        logger.info("OlfactoryCircuit initialized")
    
    def step(self, odor_input: float, dt: float = 0.001) -> Tuple[np.ndarray, Dict]:
        """
        Execute one time step of the neural circuit.
        
        Args:
            odor_input: Odor concentration (0.0 to 1.0)
            dt: Integration timestep (seconds)
            
        Returns:
            Tuple of (DN activity, spike dictionary)
        """
        
        # 1. ORN layer - converts odor to neural activity
        self._update_orn(odor_input, dt)
        
        # 2. PN layer - first processing stage
        self._update_layer(
            self.pn_membrane, self.pn_spikes,
            self.orn_spikes.astype(float), self.w_orn_pn,
            dt, layer_name='PN'
        )
        
        # 3. KC layer - associative learning (SPARSE activation)
        pn_input = self.pn_spikes.astype(float)
        kc_input = np.dot(self.w_pn_kc, pn_input)
        
        # KC sparseness: only ~5% of KCs should be active at a time
        kc_threshold = np.percentile(kc_input, 95)
        kc_input[kc_input < kc_threshold] = -80  # Set below threshold to resting
        
        self._update_layer(
            self.kc_membrane, self.kc_spikes,
            pn_input * 0.1, self.w_pn_kc,  # Reduced weights
            dt, layer_name='KC', sparse=True
        )
        
        # 4. MBON layer - valence encoding
        self._update_layer(
            self.mbon_membrane, self.mbon_spikes,
            self.kc_spikes.astype(float), self.w_kc_mbon,
            dt, layer_name='MBON'
        )
        
        # 5. DN layer - motor commands (THIS IS THE OUTPUT!)
        mbon_input = self.mbon_spikes.astype(float)
        dn_input = np.dot(self.w_mbon_dn, mbon_input)
        
        # DN receives both excitation and baseline tonic firing
        dn_baseline_input = np.random.normal(10, 2, self.n_dn)  # Tonic drive
        dn_total_input = dn_input + dn_baseline_input
        
        self._update_layer(
            self.dn_membrane, self.dn_spikes,
            dn_total_input,  # DN input
            np.eye(self.n_dn) * 0.5,  # Identity matrix as "weights"
            dt, layer_name='DN'
        )
        
        # Collect spikes for logging
        spikes = {
            'orn': self.orn_spikes.copy(),
            'pn': self.pn_spikes.copy(),
            'kc': self.kc_spikes.copy(),
            'mbon': self.mbon_spikes.copy(),
            'dn': self.dn_spikes.copy()
        }
        
        # Return DN firing rates as motor output (normalized)
        dn_output = (self.dn_membrane + 70) / 20  # Normalize to ~0-1 range
        dn_output = np.clip(dn_output, -1, 1)
        
        return dn_output, spikes
    
    def _update_orn(self, odor_input: float, dt: float):
        """
        Update ORN (sensory) layer.
        Each ORN responds proportionally to odor input.
        """
        # ORN tuning (some prefer this odor, others don't)
        tuning_curves = self.config.get('tuning_curves', 
                                       np.random.uniform(0.3, 1.0, self.n_orn))
        
        # Current input (proportional to odor)
        i_input = odor_input * tuning_curves * 100  # Arbitrary scaling
        
        # Add background noise
        i_input += np.random.normal(0, 5, self.n_orn)
        
        # Update membrane potential (exponential decay)
        self.orn_membrane += (-self.orn_membrane + self.rest_potential + i_input) * dt / self.tau_m
        
        # Spike generation
        self.orn_spikes = self.orn_membrane > self.spike_threshold
        
        # Reset after spike
        self.orn_membrane[self.orn_spikes] = self.rest_potential
    
    def _update_layer(self, membrane, spikes, input_activity, weights,
                     dt, layer_name='', sparse=False):
        """
        Update a generic layer of spiking neurons.
        
        Args:
            membrane: Membrane potential vector
            spikes: Spike output vector
            input_activity: Input from previous layer
            weights: Synaptic weight matrix
            dt: Timestep
            layer_name: Name for logging
            sparse: If True, only sparsely activate neurons
        """
        
        # Synaptic input
        i_syn = np.dot(weights, input_activity)
        
        # Add lateral inhibition
        if layer_name in ['KC']:  # Dense connectivity in KC requires inhibition
            active_kc = np.sum(spikes)
            if active_kc > 0:
                i_syn -= self.lateral_inhibition * active_kc
        
        # Add spontaneous activity (noise)
        i_syn += np.random.normal(0, self.spontaneous_activity, len(membrane))
        
        # Update membrane potential
        membrane += (-membrane + self.rest_potential + i_syn) * dt / self.tau_m
        
        # Spike generation
        spikes[:] = membrane > self.spike_threshold
        
        # Reset after spike
        membrane[spikes] = self.rest_potential
    
    def apply_learning(self, reward_signal: float, learning_rate: float = 0.01):
        """
        Apply simple Hebbian learning rule.
        Strengthens weights between neurons that fire together before and after reward.
        
        Args:
            reward_signal: Reward value (-1 to +1)
            learning_rate: Learning rate
        """
        if abs(reward_signal) < 0.01:
            return  # No learning without reward
        
        # KC-MBON Hebbian learning
        kc_activity = self.kc_spikes.astype(float)
        mbon_activity = self.mbon_spikes.astype(float)
        
        # Pre-post correlation
        correlation = np.outer(mbon_activity, kc_activity)
        
        # Update weights
        self.w_kc_mbon += learning_rate * reward_signal * correlation
        
        # Keep weights bounded
        self.w_kc_mbon = np.clip(self.w_kc_mbon, -1, 1)
