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


def modify_xml_floor(xml_content: str, floor_enabled: bool, floor_config: dict) -> str:
    """
    Modifica el XML para agregar o quitar el suelo según configuración
    
    Args:
        xml_content: Contenido del XML
        floor_enabled: Si mostrar suelo o no
        floor_config: Dict con parámetros del suelo (size, color, pattern, etc.)
        
    Returns:
        XML modificado
    """
    if not floor_enabled:
        # Remover geom del suelo si existe
        xml_content = re.sub(
            r'<geom[^>]*name="floor"[^>]*>.*?</geom>',
            '',
            xml_content,
            flags=re.DOTALL
        )
        xml_content = re.sub(
            r'<geom[^>]*name="floor"[^>]*/?>',
            '',
            xml_content
        )
        return xml_content
    
    # Si floor_enabled = True, asegurarse de que existe
    size_x, size_y, size_z = floor_config.get('size', (50.0, 50.0, 0.05))
    color = floor_config.get('color', (0.8, 0.9, 1.0, 1.0))
    pattern = floor_config.get('pattern', 'grid')
    reflection = floor_config.get('reflection', 0.1)
    
    # Crear textura y material para el suelo
    floor_asset = f'''
    <!-- Textura y material del suelo -->
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512" 
             rgb1=".7 .7 .7" rgb2=".9 .9 .9"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 10" 
              texuniform="true" reflectance="{reflection}"/>
'''
    
    # Agregar textura a assets si no existe
    if '<asset>' in xml_content:
        if 'name="floor_tex"' not in xml_content:
            xml_content = xml_content.replace(
                '</asset>',
                f'{floor_asset}\n  </asset>',
                1
            )
    else:
        # Crear sección asset
        xml_content = xml_content.replace(
            '<worldbody>',
            f'<asset>{floor_asset}\n  </asset>\n  <worldbody>',
            1
        )
    
    # Crear geom del suelo
    floor_geom = f'''
    <!-- Suelo con grid de cuadrados -->
    <geom name="floor" size="{size_x} {size_y} {size_z}" type="plane" 
          material="floor_mat" rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
'''
    
    # Verificar si ya existe un suelo
    if 'name="floor"' in xml_content:
        # Reemplazar suelo existente
        xml_content = re.sub(
            r'<geom[^>]*name="floor"[^>]*/?>',
            floor_geom.strip(),
            xml_content
        )
    else:
        # Agregar suelo después de <worldbody>
        xml_content = xml_content.replace(
            '<worldbody>',
            f'<worldbody>\n    {floor_geom}',
            1
        )
    
    return xml_content


def load_and_setup_model(
    model_path: Path, 
    width: int = 1920, 
    height: int = 1080,
    environment_config = None
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Carga el modelo XML y prepara los datos de MuJoCo
    
    Args:
        model_path: Ruta al XML
        width: Resolución ancho
        height: Resolución alto
        environment_config: EnvironmentConfig con configuración del ambiente
        
    Returns:
        Tupla (model, data)
    """
    # Leer XML original
    with open(model_path, 'r') as f:
        xml_content = f.read()
    
    # Modificar para alta resolución
    if f'offwidth="{width}"' not in xml_content:
        xml_content = modify_xml_for_high_res(xml_content, width, height)
    
    # Modificar suelo según configuración
    if environment_config is not None:
        floor_config = {
            'size': environment_config.floor_size,
            'color': environment_config.floor_color,
            'pattern': environment_config.floor_pattern,
            'reflection': environment_config.floor_reflection,
        }
        xml_content = modify_xml_floor(
            xml_content,
            environment_config.floor_enabled,
            floor_config
        )
    
    # Guardar versión modificada
    modified_path = model_path.parent / "modified_neuromechfly.xml"
    with open(modified_path, 'w') as f:
        f.write(xml_content)
    
    print(f"  ✓ XML modificado guardado en: {modified_path.name}")
    
    # Cargar modelo
    model = mujoco.MjModel.from_xml_path(str(modified_path))
    data = mujoco.MjData(model)
    
    return model, data


def create_minimal_model(
    output_dir: Path, 
    environment_config = None
) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Crea un modelo MuJoCo minimal si no se encuentra el original
    
    Args:
        output_dir: Directorio donde guardar el XML
        environment_config: EnvironmentConfig con configuración del ambiente
        
    Returns:
        Tupla (model, data)
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_xml_path = output_dir / "minimal_neuromechfly.xml"
    
    # Configuración del ambiente
    if environment_config is not None:
        floor_enabled = environment_config.floor_enabled
        floor_size = environment_config.floor_size
        floor_color = environment_config.floor_color
        floor_reflection = environment_config.floor_reflection
    else:
        floor_enabled = True
        floor_size = (50.0, 50.0, 0.05)
        floor_color = (0.8, 0.9, 1.0, 1.0)
        floor_reflection = 0.1
    
    # Plantilla mejorada del modelo
    minimal_model_xml = '''<mujoco model="neuromechfly_minimal">
  <option timestep="0.0001" integrator="RK4" gravity="0 0 -9.81"/>
  
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map znear="0.001" zfar="50"/>
  </visual>
  
  <asset>
'''
    
    # Agregar textura del suelo si está habilitado
    if floor_enabled:
        minimal_model_xml += '''    <!-- Textura de suelo con patrón de cuadrados -->
    <texture name="grid_texture" type="2d" builtin="checker" width="512" height="512" 
             rgb1=".65 .65 .65" rgb2=".85 .85 .85"/>
    <material name="grid_material" texture="grid_texture" texrepeat="20 20" 
              texuniform="true" reflectance="''' + str(floor_reflection) + '''"/>
'''
    
    minimal_model_xml += '''  </asset>
  
  <worldbody>
    <!-- Iluminación mejorada -->
    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
    <light directional="true" diffuse=".5 .5 .5" specular=".1 .1 .1" pos="5 5 3" dir="-1 -1 -1"/>
    <light directional="true" diffuse=".3 .3 .3" specular=".05 .05 .05" pos="-5 5 3" dir="1 -1 -1"/>
'''
    
    # Agregar suelo si está habilitado
    if floor_enabled:
        minimal_model_xml += f'''    
    <!-- Suelo con patrón de cuadrados grises -->
    <geom name="floor" size="{floor_size[0]} {floor_size[1]} {floor_size[2]}" 
          type="plane" material="grid_material" 
          rgba="{floor_color[0]} {floor_color[1]} {floor_color[2]} {floor_color[3]}"/>
'''
    
    minimal_model_xml += '''    
    <!-- Tórax con freejoint para libertad de movimiento -->
    <body name="Thorax" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom name="thorax" type="ellipsoid" size="0.6 0.4 0.3" rgba="0.3 0.25 0.2 1" mass="0.001"/>
      
      <!-- Cabeza -->
      <body name="Head" pos="0.7 0 0.1">
        <geom name="head" type="sphere" size="0.3" rgba="0.35 0.3 0.25 1"/>
        <!-- Ojos compuestos -->
        <geom name="eye_L" type="sphere" size="0.15" pos="0.15 0.2 0.1" rgba="0.8 0.1 0.1 0.6"/>
        <geom name="eye_R" type="sphere" size="0.15" pos="0.15 -0.2 0.1" rgba="0.8 0.1 0.1 0.6"/>
      </body>
      
      <!-- Abdomen -->
      <body name="Abdomen" pos="-0.7 0 -0.1">
        <geom name="abdomen" type="ellipsoid" size="0.5 0.35 0.25" rgba="0.35 0.3 0.25 1"/>
      </body>
'''
    
    # Agregar patas con colores distintivos
    legs_config = {
        "RF": {"pos": "0.4 -0.35 -0.1", "color": "0.9 0.3 0.2"},  # Rojo
        "RM": {"pos": "0.0 -0.4 -0.15", "color": "0.9 0.6 0.1"},  # Naranja
        "RH": {"pos": "-0.4 -0.35 -0.2", "color": "0.9 0.8 0.1"},  # Amarillo
        "LF": {"pos": "0.4 0.35 -0.1", "color": "0.2 0.6 0.9"},  # Azul
        "LM": {"pos": "0.0 0.4 -0.15", "color": "0.1 0.7 0.7"},  # Cyan
        "LH": {"pos": "-0.4 0.35 -0.2", "color": "0.6 0.4 0.8"},  # Púrpura
    }
    
    for leg, config in legs_config.items():
        minimal_model_xml += f'''
      <!-- {leg} Leg -->
      <body name="{leg}_Coxa" pos="{config['pos']}">
        <joint name="joint_{leg}Coxa_yaw" type="hinge" axis="0 0 1" range="-90 90" damping="0.01"/>
        <joint name="joint_{leg}Coxa_roll" type="hinge" axis="1 0 0" range="-45 45" damping="0.01"/>
        <geom name="{leg}_coxa" type="capsule" fromto="0 0 0 0.3 0 -0.1" size="0.08" 
              rgba="{config['color']} 1"/>
        
        <body name="{leg}_Femur" pos="0.3 0 -0.1">
          <joint name="joint_{leg}Femur" type="hinge" axis="0 1 0" range="-120 120" damping="0.01"/>
          <joint name="joint_{leg}Femur_roll" type="hinge" axis="1 0 0" range="-30 30" damping="0.01"/>
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
              <!-- Pie con contacto -->
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
        for joint_type in ["Coxa_yaw", "Coxa_roll", "Femur", "Femur_roll", "Tibia", "Tarsus1"]:
            minimal_model_xml += f'''    <position name="act_{leg}_{joint_type}" joint="joint_{leg}{joint_type}" 
              kp="10" kv="1" forcerange="-100 100"/>\n'''
    
    minimal_model_xml += '''  </actuator>
</mujoco>'''
    
    # Guardar y cargar
    with open(model_xml_path, 'w') as f:
        f.write(minimal_model_xml)
    
    print(f"  ✓ Modelo minimal creado con suelo {'habilitado' if floor_enabled else 'deshabilitado'}")
    
    return load_and_setup_model(model_xml_path, environment_config=environment_config)