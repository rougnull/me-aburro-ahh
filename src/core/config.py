"""
Configuración centralizada para el sistema de renderizado
Permite customizar todo aspecto del render sin tocar código
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from pathlib import Path


@dataclass
class CameraConfig:
    """Configuración de cámara 3D - MOLDEABLE ✨"""
    type: str = "free"  # "free", "tracking", "fixed"
    distance: float = 12.0
    elevation: float = -15.0
    azimuth_start: float = 45.0
    azimuth_rotate: float = 0.5  # Rotación por frame (grados)
    lookat_body: str = "Thorax"  # Qué cuerpo seguir
    
    # Lookups específicas
    PRESETS = {
        "side_view": {"distance": 12.0, "elevation": -15, "azimuth_start": 90, "azimuth_rotate": 0},
        "top_view": {"distance": 15.0, "elevation": -60, "azimuth_start": 90, "azimuth_rotate": 0},
        "rotating": {"distance": 12.0, "elevation": -20, "azimuth_start": 45, "azimuth_rotate": 0.5},
        "close_up": {"distance": 8.0, "elevation": -10, "azimuth_start": 90, "azimuth_rotate": 0.3},
        "iso_view": {"distance": 14.0, "elevation": 30, "azimuth_start": 45, "azimuth_rotate": 0.2},
    }
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "CameraConfig":
        """Cargar preset de cámara"""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Preset {preset_name} no encontrado. Disponibles: {list(cls.PRESETS.keys())}")
        return cls(**cls.PRESETS[preset_name])


@dataclass
class EnvironmentConfig:
    """Configuración del ambiente (MOLDEABLE) ✨"""
    
    # Suelo
    floor_enabled: bool = True
    floor_size: Tuple[float, float, float] = (50.0, 50.0, 0.05)  # (x, y, z)
    floor_color: Tuple[float, float, float, float] = (0.8, 0.9, 1.0, 1.0)  # RGBA
    floor_pattern: str = "grid"  # "grid", "solid", "checkerboard"
    floor_reflection: float = 0.1
    
    # Iluminación
    ambient_light: Tuple[float, float, float] = (0.3, 0.3, 0.3)  # RGB
    main_light_diffuse: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    fill_light_diffuse: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    
    # Cielo/Background
    sky_enabled: bool = True
    sky_color_top: Tuple[float, float, float] = (0.4, 0.6, 0.8)
    sky_color_bottom: Tuple[float, float, float] = (0.1, 0.2, 0.3)
    
    # Gravedad
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class PartHighlightConfig:
    """Configuración para destacar partes"""
    leg: Optional[str] = None  # Resaltar una pata: "RF", "RM", etc. None = todas visibles
    segment: Optional[str] = None  # "Coxa", "Femur", "Tibia", "Tarsus" - None = todos
    highlight_opacity: float = 1.0  # Opacidad de partes destacadas
    shadow_opacity: float = 0.3  # Opacidad de partes NO destacadas
    highlight_color_override: Optional[Tuple[float, float, float]] = None  # Forzar color específico


@dataclass
class LegColorConfig:
    """Configuración de colores para patas"""
    colors: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "RF": (0.9, 0.3, 0.2, 1.0),  # Rojo
            "RM": (0.9, 0.6, 0.1, 1.0),  # Naranja
            "RH": (0.9, 0.8, 0.1, 1.0),  # Amarillo
            "LF": (0.2, 0.6, 0.9, 1.0),  # Azul
            "LM": (0.1, 0.7, 0.7, 1.0),  # Cyan
            "LH": (0.6, 0.4, 0.8, 1.0),  # Púrpura
        }
    )
    
    def get_color(self, leg: str) -> Tuple[float, float, float, float]:
        """Obtener color de pata"""
        return self.colors.get(leg, (0.5, 0.5, 0.5, 1.0))


@dataclass
class RenderConfig:
    """Configuración principal de renderizado"""
    
    # Archivos
    output_dir: Path = Path("./outputs/kinematic_replay")
    data_file: Path = Path("./data/inverse_kinematics/leg_joint_angles.pkl")
    
    # Video
    fps: int = 30
    subsample: int = 5  # Renderizar 1 de cada N frames
    width: int = 1920
    height: int = 1080
    codec: str = "libx264"
    quality: int = 9
    
    # Configuraciones modulares
    camera: CameraConfig = field(default_factory=CameraConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    leg_colors: LegColorConfig = field(default_factory=LegColorConfig)
    highlight: PartHighlightConfig = field(default_factory=PartHighlightConfig)
    
    def __post_init__(self):
        """Validar rutas"""
        self.output_dir.mkdir(exist_ok=True, parents=True)


# Shortcuts para generar configs moldeables
def create_moldeable_render(
    camera_preset: str = "rotating",
    floor_enabled: bool = True,
    highlight_leg: Optional[str] = None,
    highlight_segment: Optional[str] = None,
    custom_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
) -> RenderConfig:
    """
    Factory para crear renders moldeables fácilmente
    
    Ejemplos:
        # Render básico
        config = create_moldeable_render()
        
        # Destacar pata RF moviendo
        config = create_moldeable_render(highlight_leg="RF")
        
        # Vista lateral sin suelo
        config = create_moldeable_render(camera_preset="side_view", floor_enabled=False)
    """
    config = RenderConfig()
    config.camera = CameraConfig.from_preset(camera_preset)
    config.environment.floor_enabled = floor_enabled
    config.highlight.leg = highlight_leg
    config.highlight.segment = highlight_segment
    
    if custom_colors:
        config.leg_colors.colors.update(custom_colors)
    
    return config
