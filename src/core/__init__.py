"""
Módulo core - Funcionalidad central del sistema de renderizado
"""

from .config import (
    RenderConfig,
    CameraConfig,
    EnvironmentConfig,
    PartHighlightConfig,
    LegColorConfig,
    create_moldeable_render
)

from .data import (
    load_kinematic_data,
    format_joint_data,
    get_joint_names,
    get_leg_joints,
    get_n_frames
)

from .model import (
    find_neuromechfly_model,
    load_and_setup_model,
    create_minimal_model,
    modify_xml_for_high_res
)

__all__ = [
    # Config
    'RenderConfig',
    'CameraConfig',
    'EnvironmentConfig',
    'PartHighlightConfig',
    'LegColorConfig',
    'create_moldeable_render',
    
    # Data
    'load_kinematic_data',
    'format_joint_data',
    'get_joint_names',
    'get_leg_joints',
    'get_n_frames',
    
    # Model
    'find_neuromechfly_model',
    'load_and_setup_model',
    'create_minimal_model',
    'modify_xml_for_high_res',
]