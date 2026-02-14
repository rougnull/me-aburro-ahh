# NeuroMechFly DMN Framework - Dependencies

## Core Requirements

### Python Version
- Python 3.9+
- Tested on Python 3.11

### PyTorch (Required for DMN Framework)
```bash
# CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# GPU version (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Scientific Computing
```bash
pip install numpy scipy scientific pandas scikit-learn
```

### Connectome & Network Analysis
```bash
pip install networkx h5py pyyaml
```

### Visualization (Optional but recommended)
```bash
pip install matplotlib vispy PyQt6
```

## Complete Installation

### Option 1: Manual Installation (Recommended)

```bash
# Create virtual environment
python -m venv neuromechfly_env
source neuromechfly_env/bin/activate  # On Windows: neuromechfly_env\Scripts\activate

# Install PyTorch (choose CPU or GPU version above)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy scipy scikit-learn networkx h5py pyyaml matplotlib
```

### Option 2: Using requirements.txt

```bash
# Create file and install
pip install -r requirements_dmn.txt
```

## Full Dependency List

```
# Core neural/scientific computing
torch>=2.0.0
numpy>=1.22.0
scipy>=1.10.0
scikit-learn>=1.2.0
networkx>=3.0

# Data handling
h5py>=3.8.0
pyyaml>=6.0

# Visualization
matplotlib>=3.5.0
vispy>=0.16.0
PyQt6>=6.5.0

# Optional: FlyWire connectome access (requires authentication)
caveclient>=5.3.0

# Development (optional)
pytest>=7.0.0
black>=23.0.0
pylint>=2.17.0
