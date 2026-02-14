# NeuroMechFly DMN Framework - Quick Start Guide (Phase 6)

## What is DMN?

**Differentiable Mechanical Networks** transform NeuroMechFly from a fixed-parameter simulation into a **learning system** that optimizes neural circuits via backpropagation through time (BPTT).

- **Input**: FlyWire connectome + random initial weights
- **Process**: Gradient descent over 5000-timestep episodes
- **Output**: Learned synaptic weights that solve navigation tasks
- **Goal**: Discover what mechanisms flies must use to navigate

---

## 5-Minute Setup

### Step 1: Install PyTorch (2 minutes)

```bash
# Choose ONE based on your system:

# Option A: CPU only (no GPU needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Option B: GPU (much faster, requires NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Other Dependencies (1 minute)

```bash
pip install numpy scipy scikit-learn networkx h5py pyyaml matplotlib
```

### Step 3: Verify Setup (2 minutes)

```bash
python dmn_verify_setup.py
```

Expected output:
```
✓ All checks passed! You're ready to train DMN models.

To get started:
  1. Read: DMN_README.md
  2. Run:  python dmn_train_example.py --num-episodes 20
  3. Results will be saved to: dmn_results/
```

---

## Run Your First Training (5 minutes)

### Basic Usage

```bash
# Train for 20 episodes (~2-3 minutes on CPU)
python dmn_train_example.py --num-episodes 20
```

### With Custom Parameters

```bash
# Train longer with different loss weights
python dmn_train_example.py \
  --num-episodes 50 \
  --learning-rate 0.001 \
  --nav-weight 1.0 \
  --energy-weight 0.1

# Use GPU for 10x speedup
CUDA_VISIBLE_DEVICES=0 python dmn_train_example.py --num-episodes 100
```

### Training Output

```
==============================================================
NeuroMechFly DMN Training
==============================================================

============================================================
STAGE 1: Loading Connectome
============================================================
Loaded connectome with 2150 neurons
Total synapses: 15234
Layer sizes: {'ORN': 50, 'PN': 50, 'KC': 2000, 'MBON': 50, 'DN': 10}

{... stages 2-4 ...}

============================================================
STAGE 5: Training Network with BPTT
============================================================
Episode 1/50 | Loss: 12.3456 | Nav: 10.2341 | Energy: 0.0145
Episode 2/50 | Loss: 11.8934 | Nav: 10.0234 | Energy: 0.0089
...
Episode 50/50 | Loss: 2.1234 | Nav: 1.9876 | Energy: 0.0034
Training complete!
Best loss: 1.8932

============================================================
STAGE 6: Evaluating Learned Network
============================================================
Final Trajectory Metrics:
  total_distance_traveled: 45.23
  final_distance_to_goal: 2.34
  min_distance_to_goal: 1.87
  mean_distance_to_goal: 15.42
  mean_speed: 0.0089
  max_speed: 0.0145
  exploration_area: 892.34

Neural Activity Statistics:
  orn: mean_rate=0.0950 Hz, sparsity=90.50%
  pn: mean_rate=0.0750 Hz, sparsity=92.50%
  kc: mean_rate=0.0042 Hz, sparsity=99.58%
  mbon: mean_rate=0.0380 Hz, sparsity=96.20%
  dn: mean_rate=0.0670 Hz, sparsity=93.30%

============================================================
STAGE 7: Saving Results
============================================================
Results saved to dmn_results/20240120_143022/training_results.json

✓ Training complete! Results saved to dmn_results/20240120_143022/
```

---

## Understand the Results

Results are saved to `dmn_results/TIMESTAMP/`:

### Files Generated

1. **`training_results.json`**
   - Loss history for all 50 episodes
   - Final trajectory metrics
   - Neural activity statistics
   ```json
   {
     "training_losses": [
       {
         "total": 12.3456,
         "nav_distance": 10.2341,
         "energy": 0.0145,
         ...
       },
       ...
     ],
     "final_metrics": { ... }
   }
   ```

2. **`trained_circuit.pt`**
   - Saved model checkpoint
   - Contains learned weights, optimizer state
   - Can be loaded for further training or inference

### Interpreting Loss Curves

- **Loss should decrease** over episodes (learning is working)
- **Navigation loss** dominates initially (fly hasn't learned to navigate)
- **Energy loss** stays constant (velocity scaling fixed)
- **Sparsity loss** small if KC maintain ~2% firing

### What "Good Results" Look Like

✓ Final distance to goal < 5mm (from 50mm arena)
✓ Mean distance decreases over time
✓ KC sparsity 98-99% throughout
✓ Loss curve shows clear downward trend

✗ Loss stagnates → reduce learning rate
✗ Loss explodes → learning rate too high
✗ Loss oscillates → batch size too small

---

## Explore the Framework

### 1. Examine Learned Weights

```python
import torch
from training.optimizer import DMNTrainer

# Load checkpoint
trainer = DMNTrainer(...)
trainer.load_checkpoint("dmn_results/20240120_143022/trained_circuit.pt")

# Inspect weights
for name, param in trainer.network.named_parameters():
    if param.requires_grad:
        print(f"{name}: shape={param.shape}, mean={param.mean():.4f}, "
              f"std={param.std():.4f}")

# Example output:
# w_orn_pn: shape=[50, 50], mean=0.0234, std=0.0567
# w_pn_kc: shape=[50, 2000], mean=0.0012, std=0.0089
# w_kc_mbon: shape=[2000, 50], mean=0.0456, std=0.0876
# w_mbon_dn: shape=[50, 10], mean=0.1234, std=0.2345
```

### 2. Generate New Trajectories

```python
import torch
from training.optimizer import DMNTrainer

trainer = DMNTrainer(...)
trainer.network.eval()

# Rollout with learned network
with torch.no_grad():
    trajectory = trainer.rollout_episode(
        episode_length=5000,
        goal_position=(50, 50, 0)  # Change goal location
    )

# Analyze trajectory
positions = trajectory['positions']
distances = torch.norm(positions - torch.tensor([50, 50, 0]), dim=1)

print(f"Final distance: {distances[-1]:.2f} mm")
print(f"Mean distance: {distances.mean():.2f} mm")
print(f"Min distance: {distances.min():.2f} mm")
```

### 3. Adjust Loss Weights

```python
# Emphasize energy efficiency
loss_fn.set_weights(
    nav=1.0,      # Still navigate
    energy=1.0,   # But minimize movement!
    sparse=0.01,  # Maintain sparsity
    activity=0.001
)

# Train again with energy-conscious network
losses = trainer.train_episode(num_episodes=50)
```

### 4. Use Real FlyWire Data

```python
from connectome.fetch_data import FlyWireConnectome

# Install caveclient first:
# pip install caveclient

fetcher = FlyWireConnectome()
connectome = fetcher.fetch_olfactory_circuit(
    use_synthetic=False  # ← Get real FlyWire data
)
# Requires FlyWire authentication token
```

---

## Advanced: Batch Training with Hyperparameter Search

```bash
#!/bin/bash
# Run training with multiple learning rates

for lr in 0.0001 0.001 0.01 0.1; do
    echo "Training with LR=$lr"
    python dmn_train_example.py \
      --num-episodes 50 \
      --learning-rate $lr \
      --optimizer adam
done

# Results will be in separate dmn_results/ folders
# Use timestamps to distinguish runs
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Use CPU instead
python dmn_train_example.py --num-episodes 20
# Or reduce episode length:
# Edit dmn_train_example.py: config['episode_length'] = 1000
```

### "Module not found: connectome"
```bash
# Make sure you're in the project directory
cd NeuroMechFly\ Sim/
python dmn_train_example.py
```

### "No gradients - model not learning"
```bash
# Check learning rate
--learning-rate 0.001  # Try 0.01 instead
# Check device
python -c "import torch; print(torch.cuda.is_available())"
```

### "Synthetic data only - FlyWire unavailable"
This is normal! Real FlyWire requires credentials:
- Visit https://flywire.ai/
- Request API access
- Set credentials in `connectome/fetch_data.py`

---

## Key Concepts

### 1. Surrogate Gradients
- **Problem**: Spikes are discrete (0 or 1) → non-differentiable
- **Solution**: Use smooth sigmoid in backward pass while keeping discrete spikes in forward
- **Result**: Can learn through spike-based neurons!

### 2. Backpropagation Through Time (BPPT)
- Train entire 5000-step episode as a graph
- Gradients flow backward through all timesteps
- Update weights based on task performance

### 3. Connectivity Constraints
- **Weights** are learnable parameters
- **Sparsity pattern** (which synapses exist) is fixed from connectome
- Prevents learning invalid architectures

### 4. Multi-Objective Learning
- **Navigation**: Get to food
- **Energy**: Minimize movement
- **Sparsity**: Keep KC representation sparse
- Trade-off between objectives via weight coefficients

---

## Next Steps

### Short Term (This Week)
1. ✅ Run basic training with default parameters
2. ✅ Examine loss curves and trajectory metrics
3. ✅ Try different learning rates
4. ✅ Load and inspect learned weights

### Medium Term (This Month)
1. Integrate with embodied simulation environment
2. Train on navigation + obstacle avoidance
3. Compare learned weights to real FlyWire
4. Export weights to neuromorphic hardware

### Long Term (Research)
1. Investigate what mechanisms learning discovers
2. Test predictions on real fly experiments
3. Train multi-task networks
4. Publish findings on embodied neuroscience

---

## References

- **Surrogate Gradients**: Zenke & Ganguli (2018)
- **FlyWire Connectome**: Bates et al. (2020)
- **Differentiable Biology**: Marblestone et al. (2016)
- **Neural Optimization**: Caruana et al. (2021)

---

## Get Help

1. Read detailed docs: `DMN_README.md`
2. Check code examples in module files
3. Examine `dmn_train_example.py` for complete workflow
4. See `DEPENDENCIES_DMN.md` for installation help

---

## Summary

**You now have a research tool to:**

- ✓ Load real connectome data
- ✓ Define behavioral tasks
- ✓ Train neural networks via BPTT
- ✓ Discover emergent mechanisms
- ✓ Validate against biology

**Next command**:
```bash
python dmn_train_example.py --num-episodes 20
```

**Estimated time**: 2-5 minutes (CPU) or 20-30 seconds (GPU)

---

**Happy training! 🧠🔬**
