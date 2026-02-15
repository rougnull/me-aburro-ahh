#!/usr/bin/env python3
"""
NeuroMechFly Enhanced 3D Renderer - VERSIÓN 2.0
Sistema modular y moldeable para renderizar el modelo 3D de la mosca

Este script reemplaza los anteriores render_enhanced_3d.py, render_mujoco_3d.py
y 3d_generation.py consolidándolos en una arquitectura limpia y mantenible.

USO BÁSICO:
    python render_enhanced_3d_v2.py

USO AVANZADO (CONFIG PERSONALIZADA):
    Ver ejemplos al final del archivo
"""

import sys
from pathlib import Path

# IMPORTANTE: Agregar el directorio actual al path PRIMERO
# Esto permite que los imports funcionen correctamente
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Ahora podemos importar usando la ruta completa desde la raíz
from src.core.config import RenderConfig, CameraConfig, create_moldeable_render
from src.render.mujoco_renderer import MuJoCoRenderer


def print_header():
    """Imprime encabezado"""
    print("=" * 70)
    print("  NeuroMechFly Enhanced 3D Renderer - v2.0")
    print("  Sistema modular y moldeable")
    print("=" * 70)


def example_basic_render():
    """Ejemplo 1: Render básico con configuración por defecto"""
    print("\n📋 EJEMPLO 1: Render básico (rotating view)")
    print("-" * 70)
    
    config = RenderConfig()
    config.camera = CameraConfig.from_preset("rotating")
    
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save("ejemplo1_rotating.mp4")


def example_side_view():
    """Ejemplo 2: Vista lateral sin suelo"""
    print("\n📋 EJEMPLO 2: Vista lateral sin suelo")
    print("-" * 70)
    
    config = create_moldeable_render(
        camera_preset="side_view",
        floor_enabled=False
    )
    
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save("ejemplo2_side_view.mp4")


def example_close_up():
    """Ejemplo 3: Close-up con vista iso"""
    print("\n📋 EJEMPLO 3: Close-up isométrico")
    print("-" * 70)
    
    config = create_moldeable_render(
        camera_preset="iso_view"
    )
    config.width = 1280
    config.height = 720
    config.fps = 24
    
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save("ejemplo3_iso_closeup.mp4")


def example_highlight_leg():
    """Ejemplo 4: Destacar pata derecha delantera (RF)"""
    print("\n📋 EJEMPLO 4: Destacar pata RF (derecha delantera)")
    print("-" * 70)
    
    config = create_moldeable_render(
        camera_preset="rotating",
        highlight_leg="RF"  # Solo destacar RF
    )
    
    # Personalizar colores: hacer RF más brillante
    config.leg_colors.colors["RF"] = (1.0, 0.0, 0.0, 1.0)  # Rojo puro
    
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save("ejemplo4_highlight_rf.mp4")


def example_top_down_custom():
    """Ejemplo 5: Vista superior personalizada"""
    print("\n📋 EJEMPLO 5: Vista superior sin rotación")
    print("-" * 70)
    
    config = create_moldeable_render(
        camera_preset="top_view"
    )
    
    # Personalizaciones
    config.environment.floor_enabled = True
    config.environment.floor_color = (0.9, 0.9, 0.95, 1.0)  # Gris claro
    config.fps = 30
    config.width = 1440
    config.height = 1440  # Cuadrado para vista superior
    
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save("ejemplo5_topdown.mp4")


def main():
    """Función principal"""
    print_header()
    
    # Aquí configura cuál render ejecutar
    # Descomenta el que desees
    
    # Render básico por defecto
    config = RenderConfig()
    renderer = MuJoCoRenderer(config)
    renderer.render_and_save()
    
    # Descomentar para ejecutar otros ejemplos:
    # example_side_view()
    # example_close_up()
    # example_highlight_leg()
    # example_top_down_custom()


if __name__ == "__main__":
    main()


# ============================================================================
# GUÍA DE CONFIGURACIÓN MOLDEABLE
# ============================================================================
# 
# Para crear renders personalizados, edita la configuración:
#
# 1️⃣ CÁMARAS DISPONIBLES:
#    - "side_view"      : Vista lateral fija
#    - "top_view"       : Vista superior
#    - "rotating"       : Rotación lenta (DEFAULT)
#    - "close_up"       : Acercamiento con rotación
#    - "iso_view"       : Isométrica
#
#    Uso: config.camera = CameraConfig.from_preset("side_view")
#
#
# 2️⃣ SUELO CONFIGURABLE:
#    config.environment.floor_enabled = True/False
#    config.environment.floor_size = (100, 100, 0.05)  # x, y, z
#    config.environment.floor_color = (R, G, B, A)
#
#
# 3️⃣ DESTACAR PATAS (HIGHLIGHT):
#    config.highlight.leg = "RF"  # RF, RM, RH, LF, LM, LH
#    config.highlight.segment = "Femur"  # Coxa, Femur, Tibia, Tarsus1
#    config.highlight.shadow_opacity = 0.2  # Opacidad de patas NO destacadas
#
#
# 4️⃣ COLORES PERSONALIZADOS:
#    config.leg_colors.colors["RF"] = (1.0, 0.0, 0.0, 1.0)  # Rojo puro
#    config.leg_colors.colors["LF"] = (0.0, 0.0, 1.0, 1.0)  # Azul puro
#
#
# 5️⃣ VIDEO Y RESOLUCIÓN:
#    config.width = 1920
#    config.height = 1080
#    config.fps = 30
#    config.subsample = 5  # Renderizar cada N frames
#
#
# 6️⃣ ILUMINACIÓN:
#    config.environment.ambient_light = (0.3, 0.3, 0.3)
#    config.environment.main_light_diffuse = (0.8, 0.8, 0.8)
#
#
# ============================================================================
# EJEMPLOS AVANZADOS
# ============================================================================
#
# # Crear render moldeable fácilmente
# config = create_moldeable_render(
#     camera_preset="rotating",
#     floor_enabled=True,
#     highlight_leg="RF",
#     highlight_segment="Femur",
#     custom_colors={
#         "RF": (1.0, 0.0, 0.0, 1.0),
#         "LF": (0.0, 0.0, 1.0, 1.0),
#     }
# )
#
# renderer = MuJoCoRenderer(config)
# renderer.render_and_save("mi_render_personalizado.mp4")
#
# ============================================================================