"""
Renderizador MuJoCo mejorado y moldeable
Centraliza toda la lógica de renderizado con control granular
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import mujoco
import imageio
from tqdm import tqdm

# IMPORTANTE: Asegurar que los imports funcionan
import sys
# from src.core.config import RenderConfig, CameraConfig, EnvironmentConfig, PartHighlightConfig
# from src.core.data import load_kinematic_data, format_joint_data, get_n_frames
# from src.core.model import find_neuromechfly_model, load_and_setup_model, create_minimal_model

# Asumiendo que el script se ejecuta desde la raíz o con PYTHONPATH correcto
from src.core.config import RenderConfig
from src.core.data import load_kinematic_data, format_joint_data, get_n_frames
from src.core.model import find_neuromechfly_model, load_and_setup_model, create_minimal_model


class MuJoCoRenderer:
    """
    Renderizador MuJoCo modular y moldeable ✨
    
    Características moldeables:
    - Cámaras preconfiguradas y personalizables
    - Suelo configurable (presencia, patrón, tamaño)
    - Iluminación ajustable
    - Destacado (highlight) de patas/segmentos
    - Sistema de colores flexible
    """
    
    def __init__(self, config: RenderConfig):
        """
        Inicializa el renderer
        
        Args:
            config: RenderConfig con todas las configuraciones
        """
        self.config = config
        self.model = None
        self.data = None
        self.renderer = None
        self.formatted_data = None
        self.joint_mapping = None
        self.frames = []
        
    def load_model(self) -> bool:
        """Carga el modelo MuJoCo. Retorna True si exitoso."""
        print("\n[1/5] Cargando modelo MuJoCo...")
        
        try:
            # Intentar encontrar modelo original
            model_path = find_neuromechfly_model()
            
            if model_path is None:
                print("  ⚠️  Modelo de NeuroMechFly no encontrado, creando modelo minimal...")
                self.model, self.data = create_minimal_model(self.config.output_dir, self.config.environment)
            else:
                print(f"  ✓ Modelo encontrado: {model_path.name}")
                self.model, self.data = load_and_setup_model(
                    model_path,
                    self.config.width,
                    self.config.height,
                    self.config.environment
                )
            
            print(f"  ✓ Modelo cargado:")
            print(f"    - Cuerpos: {self.model.nbody}")
            print(f"    - Articulaciones: {self.model.njnt}")
            print(f"    - Geometrías: {self.model.ngeom}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_data(self) -> bool:
        """Carga datos cinemáticos. Retorna True si exitoso."""
        print("\n[2/5] Cargando datos cinemáticos...")
        
        try:
            raw_data = load_kinematic_data(self.config.data_file)
            self.formatted_data = format_joint_data(raw_data, self.config.subsample)
            
            n_frames = get_n_frames(self.formatted_data)
            print(f"  ✓ Datos cargados: {n_frames} frames")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error cargando datos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def map_joints(self) -> bool:
        """Mapea nombres de joints a índices MuJoCo. Retorna True si exitoso."""
        print("\n[3/5] Mapeando articulaciones...")
        
        try:
            # Crear mapeo de nombres a índices
            joint_name_map = {}
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and joint_name != "root":
                    joint_name_map[joint_name] = i
            
            # Mapear datos a índices MuJoCo
            self.joint_mapping = {}
            for joint_name in self.formatted_data.keys():
                if joint_name in joint_name_map:
                    self.joint_mapping[joint_name] = joint_name_map[joint_name]
            
            print(f"  ✓ Mapeadas {len(self.joint_mapping)}/{len(self.formatted_data)} articulaciones")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error mapeando joints: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_environment_config(self):
        """
        Aplica configuración del ambiente (iluminación, etc.)
        """
        # Configuración del ambiente se aplica principalmente a través del XML
        pass
    
    def _setup_renderer(self):
        """Configura el renderer MuJoCo"""
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.height,
            width=self.config.width
        )
    
    def _setup_scene_option(self) -> mujoco.MjvOption:
        """Configura opciones de escena"""
        scene_option = mujoco.MjvOption()
        
        # Mostrar puntos de contacto y fuerzas
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False # Desactivado por limpieza visual
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        
        return scene_option
    
    def _setup_camera(self) -> mujoco.MjvCamera:
        """Configura cámara según configuración inicial"""
        camera = mujoco.MjvCamera()
        cam_config = self.config.camera
        
        if cam_config.type == "free":
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        elif cam_config.type == "tracking":
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cam_config.lookat_body)
                if body_id >= 0:
                    camera.trackbodyid = body_id
            except:
                pass 
        else:  # fixed
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        camera.distance = cam_config.distance
        camera.elevation = cam_config.elevation
        camera.azimuth = cam_config.azimuth_start
        
        return camera
    
    def render(self) -> bool:
        """
        Renderiza la animación completa
        
        Returns:
            True si se completó exitosamente
        """
        print("\n[4/5] Renderizando video 3D...")
        
        try:
            self._setup_renderer()
            scene_option = self._setup_scene_option()
            camera = self._setup_camera()
            
            # Preparar datos
            n_frames = get_n_frames(self.formatted_data)
            render_indices = range(0, n_frames, self.config.subsample)
            render_indices = list(render_indices)
            
            print(f"  Renderizando {len(render_indices)} frames ({self.config.fps} FPS)...")
            
            # Resetear simulación
            mujoco.mj_resetData(self.model, self.data)
            if self.model.nq >= 7:
                self.data.qpos[0:3] = [0.0, 0.0, 1.5]  # Posición inicial
                self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Orientación
            
            # Identificar ID del cuerpo principal para seguimiento (Thorax)
            thorax_id = -1
            try:
                thorax_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Thorax")
            except:
                pass

            # Renderizar frames
            with tqdm(enumerate(render_indices), total=len(render_indices), desc="  Progreso") as pbar:
                for idx, frame_idx in pbar:
                    # Aplicar ángulos de joints
                    for joint_name, mujoco_idx in self.joint_mapping.items():
                        angle = self.formatted_data[joint_name][frame_idx]
                        qpos_addr = self.model.jnt_qposadr[mujoco_idx]
                        self.data.qpos[qpos_addr] = angle
                    
                    # Avanzar simulación física
                    mujoco.mj_step(self.model, self.data)
                    
                    # --- LÓGICA DE CÁMARA FOLLOW FLY CORREGIDA ---
                    # Si follow_fly está activo, actualizamos el lookat de la cámara
                    # independientemente de si la cámara es FIXED, FREE o ROTATING.
                    if self.config.camera.follow_fly:
                        target_pos = None
                        if thorax_id >= 0:
                            target_pos = self.data.xpos[thorax_id].copy()
                        else:
                            # Fallback al cuerpo 1 o root qpos
                            target_pos = self.data.qpos[:3].copy()
                        
                        camera.lookat = target_pos
                    # ----------------------------------------------
                    
                    # Rotación de cámara si está configurada (se suma al azimuth actual)
                    camera.azimuth = self.config.camera.azimuth_start + idx * self.config.camera.azimuth_rotate
                    
                    # Renderizar
                    self.renderer.update_scene(self.data, camera, scene_option)
                    pixels = self.renderer.render()
                    self.frames.append(pixels)
            
            print(f"  ✓ {len(self.frames)} frames renderizados")
            return True
            
        except Exception as e:
            print(f"  ✗ Error renderizando: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_video(self, output_filename: Optional[str] = None) -> bool:
        """
        Guarda los frames renderizados como video
        
        Args:
            output_filename: Nombre del archivo de salida (opcional)
            
        Returns:
            True si se guardó exitosamente
        """
        print("\n[5/5] Guardando video...")
        
        if not self.frames:
            print("  ✗ No hay frames para guardar")
            return False
        
        try:
            if output_filename is None:
                camera_mode = self.config.camera.type
                output_filename = f"neuromechfly_3d_{camera_mode}.mp4"
            
            output_path = self.config.output_dir / output_filename
            
            # Intentar codec h264
            try:
                writer = imageio.get_writer(
                    output_path,
                    fps=self.config.fps,
                    codec=self.config.codec,
                    quality=self.config.quality,
                    pixelformat='yuv420p',
                    macro_block_size=1
                )
                
                for frame in tqdm(self.frames, desc="  Codificando"):
                    writer.append_data(frame)
                
                writer.close()
                
            except Exception as e:
                print(f"  ⚠️  Error con codec {self.config.codec}: {e}")
                print(f"  Intentando con GIF...")
                output_path = self.config.output_dir / output_filename.replace(".mp4", ".gif")
                imageio.mimsave(output_path, self.frames, fps=self.config.fps)
            
            # Mostrar información
            file_size = output_path.stat().st_size / (1024 * 1024)
            duration = len(self.frames) / self.config.fps
            
            print(f"\n  ✓ Video guardado: {output_path.name}")
            print(f"    Tamaño: {file_size:.2f} MB")
            print(f"    Duración: {duration:.1f}s ({len(self.frames)} frames)")
            print(f"    Resolución: {self.config.width}x{self.config.height} @ {self.config.fps} FPS")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Error guardando video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def render_and_save(self, output_filename: Optional[str] = None) -> bool:
        """
        Pipeline completo: carga -> renderiza -> guarda
        
        Returns:
            True si todo fue exitoso
        """
        success = (
            self.load_model() and
            self.load_data() and
            self.map_joints() and
            self.render() and
            self.save_video(output_filename)
        )
        
        if success:
            print("\n" + "=" * 70)
            print("✓ RENDERIZADO COMPLETO!")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ ERROR DURANTE RENDERIZADO")
            print("=" * 70)
        
        return success