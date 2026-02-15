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
    follow_fly: bool = True  # Si la cámara sigue a la mosca (suelo se mueve con ella)
    
    # Lookups específicas
    PRESETS = {
        "side_view": {"distance": 12.0, "elevation": -15, "azimuth_start": 90, "azimuth_rotate": 0, "follow_fly": True},
        "top_view": {"distance": 15.0, "elevation": -60, "azimuth_start": 90, "azimuth_rotate": 0, "follow_fly": True},
        "rotating": {"distance": 12.0, "elevation": -20, "azimuth_start": 45, "azimuth_rotate": 0.5, "follow_fly": True},
        "close_up": {"distance": 8.0, "elevation": -10, "azimuth_start": 90, "azimuth_rotate": 0.3, "follow_fly": True},
        "iso_view": {"distance": 14.0, "elevation": 30, "azimuth_start": 45, "azimuth_rotate": 0.2, "follow_fly": True},
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
    """Configuración para destacar partes (MOLDEABLE) ✨"""
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
    follow_fly: bool = True,
    floor_enabled: bool = True,
    floor_size: Optional[Tuple[float, float, float]] = None,
    floor_color: Optional[Tuple[float, float, float, float]] = None,
    floor_pattern: Optional[str] = None,
    floor_reflection: Optional[float] = None,
    highlight_leg: Optional[str] = None,
    highlight_segment: Optional[str] = None,
    custom_colors: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    fps: Optional[int] = None,
    subsample: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> RenderConfig:
    """
    Factory para crear renders moldeables fácilmente
    
    Args:
        camera_preset: Preset de cámara ("rotating", "side_view", "top_view", etc.)
        follow_fly: Si la cámara sigue a la mosca (suelo se mueve con ella)
        floor_enabled: Si mostrar el suelo
        floor_size: Tamaño del suelo (x, y, z) en mm
        floor_color: Color del suelo (R, G, B, A)
        floor_pattern: Patrón del suelo ("grid", "checkerboard", "solid")
        floor_reflection: Reflectancia del suelo (0.0 a 1.0)
        highlight_leg: Pata a destacar ("RF", "RM", "RH", "LF", "LM", "LH")
        highlight_segment: Segmento a destacar ("Coxa", "Femur", "Tibia", "Tarsus1")
        custom_colors: Diccionario con colores personalizados por pata
        fps: Frames por segundo del video
        subsample: Renderizar cada N frames
        width: Ancho del video en píxeles
        height: Alto del video en píxeles
    
    Ejemplos:
        # Render básico con suelo que sigue a la mosca
        config = create_moldeable_render()
        
        # Vista lateral con suelo gris claro
        config = create_moldeable_render(
            camera_preset="side_view",
            floor_color=(0.9, 0.9, 0.95, 1.0),
            floor_reflection=0.2
        )
        
        # Sin suelo
        config = create_moldeable_render(
            floor_enabled=False
        )
        
        # Suelo más grande y brillante
        config = create_moldeable_render(
            floor_size=(100, 100, 0.1),
            floor_reflection=0.3
        )
    """
    config = RenderConfig()
    
    # Configuración de cámara
    config.camera = CameraConfig.from_preset(camera_preset)
    config.camera.follow_fly = follow_fly
    
    # Configuración de suelo
    config.environment.floor_enabled = floor_enabled
    if floor_size is not None:
        config.environment.floor_size = floor_size
    if floor_color is not None:
        config.environment.floor_color = floor_color
    if floor_pattern is not None:
        config.environment.floor_pattern = floor_pattern
    if floor_reflection is not None:
        config.environment.floor_reflection = floor_reflection
    
    # Highlight
    config.highlight.leg = highlight_leg
    config.highlight.segment = highlight_segment
    
    # Colores personalizados
    if custom_colors:
        config.leg_colors.colors.update(custom_colors)
    
    # Video settings
    if fps is not None:
        config.fps = fps
    if subsample is not None:
        config.subsample = subsample
    if width is not None:
        config.width = width
    if height is not None:
        config.height = height
    
    return config