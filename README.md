# NeuroMechFly Enhanced 3D Renderer v2.0

Sistema modular y moldeable para renderizar el modelo 3D de la mosca NeuroMechFly.

## 🚀 Inicio Rápido (30 segundos)

```python
from src.render.mujoco_renderer import MuJoCoRenderer
from src.core.config import RenderConfig

config = RenderConfig()
renderer = MuJoCoRenderer(config)
renderer.render_and_save()
```

Tu video estará en `outputs/kinematic_replay/neuromechfly_3d_free.mp4`

---

## 📋 Requisitos

### Requisitos del Sistema
- **GPU recomendada** para renderizado rápido con MuJoCo
- Python 3.8 - 3.11 (FlyGym no es compatible con Python 3.12+)
- ~2 GB espacio libre en disco
- ~4 GB RAM

### Dependencias Python
- MuJoCo (motor de física)
- NumPy, Matplotlib
- ffmpeg (codificación de video)
- FlyGym (opcional, para modelo real)

## 📥 Instalación

### Paso 1: Crear Entorno Virtual (Recomendado)
```bash
python3.11 -m venv .venv

# Activar entorno
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### Paso 2: Instalar Dependencias
```bash
pip install mujoco imageio numpy tqdm

# (Opcional) Para modelo real de NeuroMechFly:
pip install "flygym[examples]"
```

### Paso 3: Instalar ffmpeg
```bash
# Windows (PowerShell)
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

## 🎨 Ejemplos de Uso

### Ejemplo 1: Render Básico
```python
from src.render.mujoco_renderer import MuJoCoRenderer
from src.core.config import RenderConfig

config = RenderConfig()
renderer = MuJoCoRenderer(config)
renderer.render_and_save()
```

### Ejemplo 2: Vista Lateral Fija
```python
from src.core.config import create_moldeable_render
from src.render.mujoco_renderer import MuJoCoRenderer

config = create_moldeable_render(camera_preset="side_view")
renderer = MuJoCoRenderer(config)
renderer.render_and_save("side_view.mp4")
```

### Ejemplo 3: Destacar Pata RF
```python
config = create_moldeable_render(
    camera_preset="close_up",
    highlight_leg="RF"
)
config.leg_colors.colors["RF"] = (1.0, 0.0, 0.0, 1.0)  # Rojo
renderer = MuJoCoRenderer(config)
renderer.render_and_save("rf_highlight.mp4")
```

### Ejemplo 4: Sin Suelo + Vista Superior
```python
config = create_moldeable_render(
    camera_preset="top_view",
    floor_enabled=False
)
config.width = 1440
config.height = 1440
renderer = MuJoCoRenderer(config)
renderer.render_and_save("topdown.mp4")
```

### Ejemplo 5: Alta Calidad 4K
```python
config = create_moldeable_render(camera_preset="rotating")
config.width = 3840
config.height = 2160
config.fps = 60
config.quality = 10
renderer = MuJoCoRenderer(config)
renderer.render_and_save("4k_60fps.mp4")
```

### Ejemplo 6: Colores Personalizados
```python
config = create_moldeable_render(
    custom_colors={
        "RF": (1.0, 0.0, 0.0, 1.0),  # Rojo
        "RM": (1.0, 0.5, 0.0, 1.0),  # Naranja
        "RH": (1.0, 1.0, 0.0, 1.0),  # Amarillo
        "LF": (0.0, 0.5, 1.0, 1.0),  # Azul
        "LM": (0.0, 0.0, 1.0, 1.0),  # Azul oscuro
        "LH": (0.5, 0.0, 1.0, 1.0),  # Púrpura
    }
)
renderer = MuJoCoRenderer(config)
renderer.render_and_save("rainbow.mp4")
```

## ⚙️ Parámetros Configurables

### Cámaras Disponibles
```
"side_view"    ← Vista lateral (excelente análisis)
"top_view"     ← Vista superior (patrones de marcha)
"rotating"     ← Rotación lenta (DEFAULT)
"close_up"     ← Acercamiento (detalles)
"iso_view"     ← Isométrica (análisis técnico)
```

### Patas (Para Highlight)
```
"RF"  ← Right Front (derecha delantera)
"RM"  ← Right Middle (derecha media)
"RH"  ← Right Hind (derecha trasera)
"LF"  ← Left Front (izquierda delantera)
"LM"  ← Left Middle (izquierda media)
"LH"  ← Left Hind (izquierda trasera)
```

### Segmentos
```
"Coxa"     ← Primera articulación
"Femur"    ← Segunda articulación
"Tibia"    ← Tercera articulación
"Tarsus1"  ← Punta de pata
```

### Parámetros Principales
```python
config = RenderConfig()

# Video
config.width = 1920               # Ancho (píxeles)
config.height = 1080              # Alto (píxeles)
config.fps = 30                   # Frames por segundo (24, 30, 60)
config.quality = 9                # Calidad (1-10)
config.subsample = 5              # 1=cada frame, 5=cada 5to

# Cámara
config.camera.distance = 12.0     # Distancia
config.camera.elevation = -15     # Ángulo vertical
config.camera.azimuth_start = 45  # Ángulo horizontal
config.camera.azimuth_rotate = 0.5  # Rotación por frame

# Suelo
config.environment.floor_enabled = True
config.environment.floor_size = (50, 50, 0.05)  # (x, y, z)
config.environment.floor_color = (0.8, 0.9, 1.0, 1.0)  # RGBA

# Highlight
config.highlight.leg = "RF"       # Destacar pata
config.highlight.segment = "Femur"  # Destacar segmento
config.highlight.shadow_opacity = 0.3  # Opacidad sombra

# Iluminación
config.environment.ambient_light = (0.3, 0.3, 0.3)
config.environment.main_light_diffuse = (0.8, 0.8, 0.8)
config.environment.fill_light_diffuse = (0.4, 0.4, 0.4)

# Colores
config.leg_colors.colors["RF"] = (1.0, 0.0, 0.0, 1.0)
```

## Troubleshooting

### GPU Issues on Google Colab
If running on Google Colab, you need to:
1. Go to Runtime → Change runtime type
2. Select GPU (e.g., T4 GPU)
3. Run the notebook

### MuJoCo Rendering Issues
If you see errors related to MuJoCo rendering:
```bash
# Set environment variable for CPU rendering
export MUJOCO_GL=osmesa  # Linux/Mac
set MUJOCO_GL=osmesa     # Windows
```

### Memory Issues
If you run out of RAM:
- Set `ENABLE_VISION = False` to disable vision
- Reduce simulation time by editing the data

### Import Errors
If you get `ModuleNotFoundError: No module named 'flygym'`:
```bash
# Make sure you're in the correct virtual environment
# Then reinstall:
pip install --upgrade "flygym[examples]"
```

## What is Kinematic Replay?

Kinematic replay is a technique where:
1. Real fly movements are recorded using high-speed cameras
2. 3D poses are reconstructed from 2D video
3. Joint angles are calculated via inverse kinematics
4. These angles are fed into a physics simulation
5. The simulation predicts forces, torques, and dynamics

This allows researchers to:
- ✅ Understand biomechanics of locomotion
- ✅ Measure forces that can't be directly measured experimentally
- ✅ Test hypotheses about motor control
- ✅ Develop realistic models for robotics

## Data Source

The kinematic data comes from:
- Real *Drosophila melanogaster* walking in a corridor
- Recorded at 360 fps, downsampled to 130 fps
- 3D pose reconstruction and inverse kinematics
- Published in NeuroMechFly workshop materials

Data is automatically downloaded from:
https://github.com/NeLy-EPFL/neuromechfly-workshop

## References

### NeuroMechFly Paper
Lobato-Rios et al. (2022). "NeuroMechFly, a neuromechanical model of adult Drosophila melanogaster." *Nature Methods*, 19, 620–627.
https://doi.org/10.1038/s41592-022-01466-7

### FlyGym Documentation
https://neuromechfly.org/

### GitHub Repository
https://github.com/NeLy-EPFL/flygym

## Support

For issues with:
- **This script**: Check the code comments or modify parameters
- **FlyGym installation**: Visit https://neuromechfly.org/installation.html
- **FlyGym bugs**: Open an issue at https://github.com/NeLy-EPFL/flygym/issues
- **Scientific questions**: Contact the NeuroMechFly team

## License

This script is based on the FlyGym tutorials and examples.
FlyGym is licensed under Apache 2.0.

## Acknowledgments

This implementation is based on the official NeuroMechFly tutorial notebooks:
- 1_getting_started.ipynb
- 2_kinematic_replay.ipynb

Special thanks to the NeuroMechFly team at EPFL for creating this amazing tool!