# Differentiable Mechanical Networks (DMN) Framework

## Overview

This framework transforms NeuroMechFly from a **fixed-parameter simulation** into a **learnable neural network optimized for behavioral tasks**. Using backpropagation through time (BPTT), we can discover what neural mechanisms and synaptic weights produce emergent behaviors optimal for navigation, energy efficiency, and sensorimotor control.

**Key Innovation**: Combine differentiable neural dynamics (PyTorch) with real connectome data (FlyWire) to reverse-engineer Drosophila behavior.

---

## Architecture

### Directory Structure

```
NeuroMechFly Sim/
├── connectome/                 # FlyWire connectome integration
│   ├── fetch_data.py          # Download connectivity from FlyWire/Codex
│   ├── adjacency_matrix.py    # Convert synapses to learnable weight matrices
│   ├── cell_types.yaml        # Neuron nomenclature and properties
│   └── cache/                 # Cached connectome data
│
├── simulation/
│   ├── mechanism.py           # Differentiable LIF neurons in PyTorch
│   ├── ...                    # (existing files)
│
├── training/                  # Training infrastructure
│   ├── loss_functions.py      # Task-based objectives (navigation, energy, sparsity)
│   ├── optimizer.py           # BPTT training loop with checkpointing
│   ├── checkpoints/           # Saved model weights
│   └── ...
│
└── ...
```

---

## Stage 1: Data Layer (`connectome/`)

### `fetch_data.py`

**Purpose**: Retrieve connectivity information from FlyWire connectome database.

```python
from connectome.fetch_data import FlyWireConnectome

# Initialize fetcher
fetcher = FlyWireConnectome(use_cache=True)

# Fetch synthetic olfactory circuit (or real FlyWire w/ credentials)
connectome = fetcher.fetch_olfactory_circuit(
    include_layers=['ORN', 'PN', 'KC', 'MBON', 'DN'],
    use_synthetic=False  # Set True to use real FlyWire
)

# Get connectivity info
print(f"Total neurons: {len(connectome['neurons'])}")
print(f"Total synapses: {connectome['total_synapses']}")
print(f"Layer sizes: {connectome['layer_sizes']}")
```

**Data Structure**:
- `neurons`: Dict of `NeuronMetadata` (cell type, layer, position)
- `synapses`: List of `SynapseConnection` (pre_id, post_id, count)
- `layer_sizes`: Neurons per layer
- `source`: 'synthetic' or 'flywire'

### `adjacency_matrix.py`

**Purpose**: Convert synapse counts to learnable weight matrices.

```python
from connectome.adjacency_matrix import AdjacencyMatrixGenerator

generator = AdjacencyMatrixGenerator()

# Create learnable weight matrices
weights_orn_pn, mask_orn_pn = generator.create_weight_matrix(
    connectivity_data=connectome,
    normalize=True,
    learnable=True  # Makes it a torch.nn.Parameter
)

print(f"Weight shape: {weights_orn_pn.shape}")
print(f"Sparsity: {(1 - mask_orn_pn.sum() / mask_orn_pn.numel()):.2%}")
```

**Key Features**:
- **Connectivity masks**: Preserve sparsity patterns while learning weights
- **Weight initialization**: Based on synapse counts (log-linear)
- **Normalization**: By fan-in to prevent saturating currents
- **Sparse format**: Support for very large connectomes (CSR/COO)

### `cell_types.yaml`

**Purpose**: Reference for neuron types, synaptic properties, and learning dynamics.

Contains:
- Layer descriptions (ORN, PN, KC, MBON, DN)
- Synaptic properties (time constants, reversal potentials)
- Circuit topology
- Baseline activity patterns
- Validation data from literature

---

## Stage 2: Neural Mechanism (`simulation/mechanism.py`)

**Purpose**: Implement differentiable spiking neuron dynamics compatible with PyTorch autograd.

### LIF Neuron Model

**Dynamics** (Leaky Integrate-and-Fire):

$$\tau \frac{dV}{dt} = -V + RI_{syn}$$

**Discretized**:

$$V_{t+1} = \alpha V_t + (1-\alpha) I_{syn}$$

where $\alpha = e^{-\Delta t / \tau}$ (decay factor)

**Key Innovation**: Surrogate Gradient Function
- **Forward**: Returns discrete spike (1 if V > θ, else 0)
- **Backward**: Uses smooth surrogate σ(β·V) for gradients

This allows learning with binary outputs while maintaining differentiability!

```python
from simulation.mechanism import LIFCell, DifferentiableOlfactoryCircuit

# Create single LIF population
lif = LIFCell(
    n_neurons=100,
    tau_ms=20.0,
    threshold_mv=-50.0,
    learnable_tau=True,  # Make time constant learnable
    device='cuda'
)

# Single timestep
I_syn = torch.randn(100)
spikes, V = lif(I_syn)

# Full circuit
circuit = DifferentiableOlfactoryCircuit(
    connectivity_data=connectome,
    weights_orn_pn=w_orn_pn,
    weights_pn_kc=w_pn_kc,
    weights_kc_mbon=w_kc_mbon,
    weights_mbon_dn=w_mbon_dn,
    masks={...},
    learnable=True
)

# Forward pass
dn_spikes, activations = circuit(odor_input, return_activations=True)
```

---

## Stage 3: Loss Functions (`training/loss_functions.py`)

**Purpose**: Define what we want the network to learn.

### Loss Components

1. **Navigation Loss**: Minimize distance to food source
   $$L_{nav} = ||position - goal||_2 + \text{penalty if no progress}$$

2. **Energy Loss**: Minimize movement (energy is ∝ velocity²)
   $$L_{energy} = w_{energy} \sum ||v_t||^2$$

3. **Sparsity Loss**: Maintain KC sparsity (~2% active)
   $$L_{sparse} = ||actual\_sparsity - 0.02||^2$$

4. **Activity Regularization**: Prevent pathological firing
   $$L_{activity} = \text{L1 spikes + penalty for firing rate violations}$$

### Total Loss

$$L_{total} = w_{nav} L_{nav} + w_{energy} L_{energy} + w_{sparse} L_{sparse} + w_{activity} L_{activity}$$

```python
from training.loss_functions import CombinedLoss, TrajectoryAnalyzer

# Create loss function
loss_fn = CombinedLoss(
    navigation_weight=1.0,
    energy_weight=0.1,
    sparsity_weight=0.01,
    activity_weight=0.001,
    goal_position=(50.0, 50.0, 0.0)
)

# Compute loss over trajectory
positions = torch.randn(5000, 3)  # (T, 3)
velocities = torch.randn(5000, 3)
spikes = {
    'kc': torch.randn(5000, 2000) > 0.98,
    'mbon': torch.randn(5000, 50) > 0.95,
    ...
}

losses = loss_fn(positions, velocities, spikes)
total = losses['total']

# Analyze trajectory
metrics = TrajectoryAnalyzer.compute_trajectory_metrics(
    positions, velocities, goal=(50, 50, 0)
)
```

---

## Stage 4: Training (`training/optimizer.py`)

**Purpose**: Backpropagation through time for end-to-end learning.

### BPTT Algorithm

```
for episode in range(num_episodes):
    # 1. Rollout: execute network over T timesteps
    trajectory = network.rollout(episode_length=5000)
    
    # 2. Compute loss
    loss = loss_function(trajectory)
    
    # 3. Backward pass (automatic differentiation through time)
    loss.backward()  # ← Gradients flow back through 5000 timesteps!
    
    # 4. Gradient clipping (prevent explosion)
    clip_grad_norm(network.parameters())
    
    # 5. Update parameters
    optimizer.step()
```

**Key Feature**: PyTorch's autograd automatically backpropagates through:
- All LIF neuron timesteps
- All synaptic connections
- All layer compositions

```python
from training.optimizer import DMNTrainer, TrainingConfig

# Setup training
config = TrainingConfig(
    learning_rate=0.001,
    optimizer='adam',
    num_episodes=100,
    episode_length_steps=5000,
    nav_weight=1.0,
    energy_weight=0.1
)

trainer = DMNTrainer(circuit, loss_fn, config, environment=env)

# Train!
losses = trainer.train_episode(num_episodes=100)

# Save best model
trainer.save_checkpoint("best_model.pt")
```

---

## Complete Workflow Example

```python
import torch
from connectome.fetch_data import FlyWireConnectome
from connectome.adjacency_matrix import AdjacencyMatrixGenerator
from simulation.mechanism import DifferentiableOlfactoryCircuit
from training.loss_functions import CombinedLoss
from training.optimizer import DMNTrainer, TrainingConfig

# ============ STAGE 1: Load Connectome ============
print("Loading FlyWire connectome...")
fetcher = FlyWireConnectome(use_cache=True)
connectome = fetcher.fetch_olfactory_circuit(
    include_layers=['ORN', 'PN', 'KC', 'MBON', 'DN'],
    use_synthetic=True  # Use False for real FlyWire
)

# ============ STAGE 2: Create Weight Matrices ============
print("Generating weight matrices...")
generator = AdjacencyMatrixGenerator()

matrices = {}
masks = {}
layer_pairs = [
    ('orn', 'pn'), ('pn', 'kc'), ('kc', 'mbon'), ('mbon', 'dn')
]

for src, dst in layer_pairs:
    weights, mask = generator.create_weight_matrix(
        connectome, normalize=True, learnable=True
    )
    matrices[f'w_{src}_{dst}'] = weights
    masks[f'mask_{src}_{dst}'] = mask

# ============ STAGE 3: Build Network ============
print("Building differentiable circuit...")
circuit = DifferentiableOlfactoryCircuit(
    connectivity_data=connectome,
    weights_orn_pn=matrices['w_orn_pn'],
    weights_pn_kc=matrices['w_pn_kc'],
    weights_kc_mbon=matrices['w_kc_mbon'],
    weights_mbon_dn=matrices['w_mbon_dn'],
    masks=masks,
    learnable=True
)

# ============ STAGE 4: Setup Loss ============
print("Creating loss function...")
loss_fn = CombinedLoss(
    navigation_weight=1.0,
    energy_weight=0.1,
    sparsity_weight=0.01,
    activity_weight=0.001,
    goal_position=(50.0, 50.0, 0.0)
)

# ============ STAGE 5: Train ============
print("Training network with BPTT...")
config = TrainingConfig(
    num_episodes=100,
    learning_rate=0.001,
    optimizer='adam'
)

trainer = DMNTrainer(circuit, loss_fn, config)
losses = trainer.train_episode(num_episodes=100)

# ============ STAGE 6: Evaluate ============
print("Evaluating learned network...")
circuit.eval()
with torch.no_grad():
    trajectories = trainer.rollout_episode(episode_length=5000)
    metrics = TrajectoryAnalyzer.compute_trajectory_metrics(
        trajectories['positions'],
        trajectories['velocities'],
        goal=(50, 50, 0)
    )
    print(f"Final distance to goal: {metrics['final_distance_to_goal']:.2f} mm")
    print(f"Mean distance: {metrics['mean_distance_to_goal']:.2f} mm")
    print(f"Exploration area: {metrics['exploration_area']:.2f} mm²")

# Save model
trainer.save_checkpoint("trained_network.pt")
```

---

## Key Advantages of This Approach

| Aspect | Before (Fixed Parameters) | After (DMN Learning) |
|--------|---------------------------|---------------------|
| **Connectivity** | Random sparse (2%) | Real FlyWire connectome |
| **Learning** | Event-based STDP | Global gradient descent (BPTT) |
| **Optimization Target** | Simulation accuracy | Task performance (navigation, efficiency) |
| **Discoverable Insights** | "What does this circuit do?" | "What must this circuit learn to solve this task?" |
| **Validation** | Comparing to biology | Predicting learned weights ↔ actual synapses |

---

## Research Applications

### 1. Reverse Engineering

Learn what mechanisms (synaptic weights, time constants) are necessary for observed behaviors.

**Question**: "If we only constrain the connectome structure but allow weights to evolve, what solutions does gradient descent find?"

### 2. Counterfactual Analysis

Train networks with modified connectomes, compare learned solutions.

**Example**: Remove specific connection types → does learning compensate or fail?

### 3. Neuroethology

Discover relationship between neural circuit parameters and behavioral strategies.

**Example**: Does energy-conscious training produce different locomotion patterns?

### 4. Hardware Implementation

Learned weights can be mapped to neuromorphic hardware (Intel Loihi, BrainScaleS).

---

## Next Steps

1. **Install PyTorch**: 
   ```bash
   pip install torch torchvision torchaudio scikit-learn networkx
   ```

2. **Run first training loop**:
   ```bash
   python -m training.optimizer
   ```

3. **Connect to real FlyWire** (requires credentials):
   ```bash
   pip install caveclient
   # Set FlyWire auth token
   # Modify connectome/fetch_data.py _fetch_real_olfactory_circuit()
   ```

4. **Integrate with environment**:
   - Link to actual NeuroMechFly physics simulator
   - Or use synthetic environment in optimizer.py

5. **Analyze learned circuits**:
   - Extract learned connectivity patterns
   - Compare to FlyWire
   - Compute manifold structure with UMAP/PCA

---

## References

1. **Differentiable neural simulation**:
   - Zenke & Ganguli (2018) "Superspike" - surrogate gradients
   - Zenke et al. (2018) TCCN framework

2. **FlyWire connectome**:
   - Bates et al. (2020) "The connectome of Drosophila central brain"
   - Zheng et al. (2020) "Structured sampling of Drosophila connectome"

3. **Neural circuit learning**:
   - Caruana et al. (2021) meta-learning in neural systems
   - Marblestone et al. (2016) "Toward an integration of deep learning and neuroscience"

4. **Drosophila neuroethology**:
   - Turner et al. (2020) olfactory circuit
   - Takemura et al. (2017) synaptic connectivity rules

---

## Authors & Citation

NeuroMechFly DMN Framework - Educational & Research Use

```bibtex
@software{neuromechfly_dmn,
  title={NeuroMechFly: Differentiable Mechanical Networks},
  author={Your Name},
  year={2024},
  note={Combines embodied RL simulation with biological connectomics}
}
```

---

**Questions?** See the example code in each module file.
