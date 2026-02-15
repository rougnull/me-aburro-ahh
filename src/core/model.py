"""
Búsqueda y carga de modelos MuJoCo
Centraliza la lógica de encontrar y cargar el modelo de NeuroMechFly
"""

import re
from pathlib import Path
from typing import Optional, Tuple
import mujoco


def find_neuromechfly_model() -> Optional[Path]:
    """
    Busca el modelo XML de NeuroMechFly en el sistema
    Este fue código duplicado entre los 3 scripts - ahora centralizado ✨
    
    Busca en:
    1. flygym.data.mjcf
    2. flygym.data
    3. flygym.mujoco
    4. flygym.models
    5. Búsqueda recursiva por nombre
    
    Returns:
        Path al archivo XML encontrado, o None si no existe
    """
    try:
        import flygym
        flygym_path = Path(flygym.__file__).parent
        
        # Rutas comunes donde FlyGym almacena el modelo
        possible_paths = [
            flygym_path / "data" / "mjcf" / "neuromechfly.xml",
            flygym_path / "data" / "neuromechfly.xml",
            flygym_path / "mujoco" / "neuromechfly.xml",
            flygym_path / "models" / "neuromechfly.xml",
            flygym_path / "assets" / "neuromechfly.xml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Búsqueda recursiva si no se encontró
        for xml_file in flygym_path.rglob("*.xml"):
            if "neuromechfly" in xml_file.name.lower():
                return xml_file
        
        return None
        
    except ImportError:
        return None


def modify_xml_for_high_res(xml_content: str, width: int = 1920, height: int = 1080) -> str:
    """
    Modifica un XML de MuJoCo para aumentar la resolución del framebuffer
    
    Args:
        xml_content: Contenido del XML
        width: Ancho deseado
        height: Alto deseado
        
    Returns:
        XML modificado
    """
    # Si ya tiene configuración, reemplazar
    if '<visual>' in xml_content:
        xml_content = re.sub(
            r'<visual>.*?</visual>',
            f'<visual>\n    <global offwidth="{width}" offheight="{height}"/>\n    <quality shadowsize="4096"/>\n  </visual>',
            xml_content,
            flags=re.DOTALL
        )
    else:
        # Agregar al inicio del modelo
        xml_content = xml_content.replace(
            '<mujoco',
            f'<mujoco>\n  <visual>\n    <global offwidth="{width}" offheight="{height}"/>\n    <quality shadowsize="4096"/>\n  </visual>',
            1
        )
    
    return xml_content


def load_and_setup_model(model_path: Path, width: int = 1920, height: int = 1080) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Carga el modelo XML y prepara los datos de MuJoCo
    
    Args:
        model_path: Ruta al XML
        width: Resolución ancho
        height: Resolución alto
        
    Returns:
        Tupla (model, data)
    """
    # Si es el modelo original de neuromechfly, modificar para alta resolución
    if "neuromechfly" in str(model_path) and "modified" not in str(model_path):
        with open(model_path, 'r') as f:
            xml_content = f.read()
        
        # Comprobar si ya está modificado
        if f'offwidth="{width}"' not in xml_content:
            xml_content = modify_xml_for_high_res(xml_content, width, height)
            
            # Guardar versión modificada
            modified_path = model_path.parent / "modified_neuromechfly.xml"
            with open(modified_path, 'w') as f:
                f.write(xml_content)
            
            model_path = modified_path
    
    # Cargar modelo
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    return model, data


def create_minimal_model(output_dir: Path) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Crea un modelo MuJoCo minimal si no se encuentra el original
    Este fue código duplicado - ahora centralizado ✨
    
    Args:
        output_dir: Directorio donde guardar el XML
        
    Returns:
        Tupla (model, data)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_xml_path = output_dir / "minimal_neuromechfly.xml"
    
    # Plantilla mejorada del modelo
    minimal_model_xml = '''<mujoco model="neuromechfly_minimal">
  <option timestep="0.0001" integrator="RK4" gravity="0 0 -9.81"/>
  
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1=".2 .3 .4" rgb2=".3 .4 .5"/>
    <material name="grid" texture="grid" texrepeat="10 10" texuniform="true" reflectance=".1"/>
  </asset>
  
  <worldbody>
    <!-- Iluminación -->
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
    <light directional="true" diffuse=".4 .4 .4" specular=".1 .1 .1" pos="5 5 3" dir="-1 -1 -1"/>
    
    <!-- Suelo -->
    <geom name="floor" size="50 50 .05" type="plane" material="grid" rgba=".8 .9 1 1"/>
    
    <!-- Tórax con freejoint para libertad de movimiento -->
    <body name="Thorax" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom name="thorax" type="ellipsoid" size="0.6 0.4 0.3" rgba="0.3 0.25 0.2 1" mass="0.001"/>
      
      <!-- Cabeza -->
      <body name="Head" pos="0.7 0 0.1">
        <geom name="head" type="sphere" size="0.3" rgba="0.35 0.3 0.25 1"/>
        <geom name="eye_L" type="sphere" size="0.15" pos="0.15 0.2 0.1" rgba="0.8 0.1 0.1 0.6"/>
        <geom name="eye_R" type="sphere" size="0.15" pos="0.15 -0.2 0.1" rgba="0.8 0.1 0.1 0.6"/>
      </body>
      
      <!-- Abdomen -->
      <body name="Abdomen" pos="-0.7 0 -0.1">
        <geom name="abdomen" type="ellipsoid" size="0.5 0.35 0.25" rgba="0.35 0.3 0.25 1"/>
      </body>
'''
    
    # Agregar patas
    legs_config = {
        "RF": {"pos": "0.4 -0.35 -0.1", "color": "0.9 0.3 0.2"},
        "RM": {"pos": "0.0 -0.4 -0.15", "color": "0.9 0.6 0.1"},
        "RH": {"pos": "-0.4 -0.35 -0.2", "color": "0.9 0.8 0.1"},
        "LF": {"pos": "0.4 0.35 -0.1", "color": "0.2 0.6 0.9"},
        "LM": {"pos": "0.0 0.4 -0.15", "color": "0.1 0.7 0.7"},
        "LH": {"pos": "-0.4 0.35 -0.2", "color": "0.6 0.4 0.8"},
    }
    
    for leg, config in legs_config.items():
        minimal_model_xml += f'''
      <!-- {leg} Leg -->
      <body name="{leg}_Coxa" pos="{config['pos']}">
        <joint name="joint_{leg}Coxa_yaw" type="hinge" axis="0 0 1" range="-90 90" damping="0.01"/>
        <geom name="{leg}_coxa" type="capsule" fromto="0 0 0 0.3 0 -0.1" size="0.08" 
              rgba="{config['color']} 1"/>
        
        <body name="{leg}_Femur" pos="0.3 0 -0.1">
          <joint name="joint_{leg}Femur" type="hinge" axis="0 1 0" range="-120 120" damping="0.01"/>
          <geom name="{leg}_femur" type="capsule" fromto="0 0 0 0.5 0 -0.3" size="0.07" 
                rgba="{config['color']} 1"/>
          
          <body name="{leg}_Tibia" pos="0.5 0 -0.3">
            <joint name="joint_{leg}Tibia" type="hinge" axis="0 1 0" range="-150 0" damping="0.01"/>
            <geom name="{leg}_tibia" type="capsule" fromto="0 0 0 0.4 0 -0.4" size="0.06" 
                  rgba="{config['color']} 1"/>
            
            <body name="{leg}_Tarsus" pos="0.4 0 -0.4">
              <joint name="joint_{leg}Tarsus1" type="hinge" axis="0 1 0" range="-90 45" damping="0.01"/>
              <geom name="{leg}_tarsus" type="capsule" fromto="0 0 0 0.3 0 -0.2" size="0.05" 
                    rgba="{config['color']} 1"/>
              <geom name="{leg}_foot" type="sphere" pos="0.3 0 -0.2" size="0.08" 
                    rgba="0.1 0.1 0.1 1" friction="1.5 0.1 0.1"/>
            </body>
          </body>
        </body>
      </body>
'''
    
    minimal_model_xml += '''
    </body>
  </worldbody>
  
  <actuator>
'''
    
    # Agregar actuadores
    for leg in legs_config.keys():
        for joint_type in ["Coxa_yaw", "Femur", "Tibia", "Tarsus1"]:
            minimal_model_xml += f'''    <position name="act_{leg}_{joint_type}" joint="joint_{leg}{joint_type}" 
              kp="10" kv="1" forcerange="-100 100"/>\n'''
    
    minimal_model_xml += '''  </actuator>
</mujoco>'''
    
    # Guardar y cargar
    with open(model_xml_path, 'w') as f:
        f.write(minimal_model_xml)
    
    return load_and_setup_model(model_xml_path)
