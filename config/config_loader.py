"""Configuration loading utilities for NeuroMechFly."""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f) or {}
    return config


def load_default_config() -> dict:
    """Load default configuration from environment.yaml."""
    config_dir = Path(__file__).parent
    
    # Load individual configs
    env_config = load_config(str(config_dir / 'environment.yaml')) or {}
    fly_config = load_config(str(config_dir / 'fly_params.yaml')) or {}
    brain_config = load_config(str(config_dir / 'brain_params.yaml')) or {}
    
    # Merge into single config
    config = {}
    config.update(env_config)
    config.update(brain_config)
    config['fly_params'] = fly_config
    
    return config
