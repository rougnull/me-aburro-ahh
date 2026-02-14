#!/usr/bin/env python3
"""
Quick Setup Verification Script for NeuroMechFly DMN Framework

Checks all dependencies and confirms system readiness for training
differentiable neural networks on the olfactory circuit.

Run: python dmn_verify_setup.py
"""

import sys
import importlib
from pathlib import Path


def check_module(module_name: str, installed_name: str = None) -> bool:
    """Check if a Python module is installed."""
    check_name = installed_name or module_name
    try:
        mod = importlib.import_module(check_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {module_name:<20} (v{version})")
        return True
    except ImportError:
        print(f"  ✗ {module_name:<20} NOT FOUND")
        return False


def check_pytorch() -> bool:
    """Verify PyTorch installation and CUDA availability."""
    print("\nPyTorch Setup:")
    try:
        import torch
        print(f"  ✓ PyTorch v{torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA version: {torch.version.cuda}")
        else:
            print(f"  ℹ No GPU detected - will use CPU (slower)")
        
        return True
    except ImportError:
        print(f"  ✗ PyTorch NOT installed")
        return False


def check_directories() -> bool:
    """Verify project directory structure."""
    print("\nProject Structure:")
    
    required_dirs = [
        'connectome',
        'training',
        'simulation',
        'analysis',
    ]
    
    all_exist = True
    for d in required_dirs:
        path = Path(d)
        if path.exists() and path.is_dir():
            print(f"  ✓ /{d}/")
        else:
            print(f"  ✗ /{d}/ NOT FOUND")
            all_exist = False
    
    return all_exist


def check_files() -> bool:
    """Verify key Python files exist."""
    print("\nCore Files:")
    
    required_files = [
        'connectome/fetch_data.py',
        'connectome/adjacency_matrix.py',
        'connectome/cell_types.yaml',
        'simulation/mechanism.py',
        'training/loss_functions.py',
        'training/optimizer.py',
        'dmn_train_example.py',
        'DMN_README.md',
    ]
    
    all_exist = True
    for f in required_files:
        path = Path(f)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {f:<40} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {f:<40} NOT FOUND")
            all_exist = False
    
    return all_exist


def main():
    """Main verification routine."""
    print("=" * 60)
    print("NeuroMechFly DMN Framework - Setup Verification")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython Version: {sys.version}")
    if sys.version_info < (3, 9):
        print("⚠ WARNING: Python 3.9+ required")
        return False
    
    # Check core dependencies
    print("\nCore Dependencies:")
    core_ok = all([
        check_module('numpy'),
        check_module('scipy'),
        check_module('torch'),
        check_module('sklearn', 'scikit-learn'),
    ])
    
    # Check PyTorch specifically
    pytorch_ok = check_pytorch()
    
    # Check framework dependencies
    print("\nFramework Dependencies:")
    fw_ok = all([
        check_module('h5py'),
        check_module('yaml', 'yaml'),
        check_module('networkx'),
    ])
    
    # Check optional visualization
    print("\nOptional Visualization:")
    all([
        check_module('matplotlib'),
        check_module('vispy'),
        check_module('PyQt6'),
    ])
    
    # Check project structure
    dir_ok = check_directories()
    file_ok = check_files()
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print("=" * 60)
    
    checks = {
        'Core dependencies': core_ok,
        'PyTorch setup': pytorch_ok,
        'Framework modules': fw_ok,
        'Directory structure': dir_ok,
        'Core files': file_ok,
    }
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:<8} {check_name}")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ All checks passed! You're ready to train DMN models.")
        print("\nTo get started:")
        print("  1. Read: DMN_README.md")
        print("  2. Run:  python dmn_train_example.py --num-episodes 20")
        print("  3. Results will be saved to: dmn_results/")
        return True
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("\n  pip install -r DEPENDENCIES_DMN.md")
        
        if not pytorch_ok:
            print("\nTo install PyTorch:")
            print("  pip install torch torchvision torchaudio --index-url \\")
            print("    https://download.pytorch.org/whl/cpu")
        
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
