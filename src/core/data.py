"""
Carga y procesamiento de datos cinemáticos
Centraliza la lógica de formateo de datos para evitar duplicación
"""

import pickle
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_kinematic_data(data_file: Path) -> Dict:
    """
    Carga datos de cinemática desde archivo pickle
    
    Args:
        data_file: Ruta al archivo pickle con datos
        
    Returns:
        Dict con datos crudos
    """
    if not data_file.exists():
        raise FileNotFoundError(f"Archivo de datos no encontrado: {data_file}")
    
    with open(data_file, "rb") as f:
        return pickle.load(f)


def format_joint_data(raw_data: Dict, subsample: int = 1) -> Dict[str, np.ndarray]:
    """
    Convierte formato de datos FlyGym (seqikpy) a formato MuJoCo interno
    
    Mapeo de segmentos:
        ThC   -> Coxa
        CTr   -> Femur
        FTi   -> Tibia
        TiTa  -> Tarsus1
    
    Args:
        raw_data: Datos crudos del archivo pickle
        subsample: Tomar cada N frames
        
    Returns:
        Dict con formato: {"joint_LEGSegment": array_ángulos, ...}
    """
    formatted = {}
    
    # Mapeo de nombres de segmentos
    segment_mapping = {
        "ThC": "Coxa",
        "CTr": "Femur",
        "FTi": "Tibia",
        "TiTa": "Tarsus1"
    }
    
    for joint, values in raw_data.items():
        # Ignorar metadatos
        if joint in ["meta", "swing_stance_time"]:
            continue
        
        # Parsear nombre de joint
        # Formato esperado: "walkerJoin_XX_LEG_SEGMENT_DOF"
        # Ejemplo: "walkerJoin_00_RF_ThC_pitch"
        try:
            # El formato real es: walkerJoin_NN_LEG_SEGMENT_DOF
            # donde NN es un número, LEG es RF/LF/etc, SEGMENT es ThC/CTr/etc
            
            # Extraer componentes
            leg = joint[6:8]  # Posiciones 6-7 son el leg code
            joint_name = joint[9:]  # Después del guión bajo
            
            # Dividir en segmento y DOF
            if "_" in joint_name:
                parts = joint_name.split("_")
                segment = parts[0]  # ThC, CTr, FTi, TiTa
                dof = parts[1] if len(parts) > 1 else "pitch"
            else:
                segment = joint_name
                dof = "pitch"
            
            # Mapear segmento
            segment_name = segment_mapping.get(segment, segment)
            
            # Crear nombre de joint en formato MuJoCo
            if dof == "pitch":
                new_key = f"joint_{leg}{segment_name}"
            else:
                new_key = f"joint_{leg}{segment_name}_{dof}"
            
            # Samplear si es necesario
            if isinstance(values, (list, np.ndarray)):
                values_sampled = np.array(values)[::subsample]
                formatted[new_key] = values_sampled
            
        except (IndexError, ValueError, KeyError) as e:
            # Skip joints que no puedan parsearse
            continue
    
    if not formatted:
        raise ValueError("No se pudieron extraer datos de joints del archivo. Verifica el formato de datos.")
    
    return formatted


def get_joint_names(formatted_data: Dict) -> List[str]:
    """Obtener nombres de joints formateados"""
    return list(formatted_data.keys())


def get_leg_joints(formatted_data: Dict, leg: str) -> Dict[str, np.ndarray]:
    """Obtener todos los joints de una pata específica"""
    prefix = f"joint_{leg}"
    return {k: v for k, v in formatted_data.items() if k.startswith(prefix)}


def get_n_frames(formatted_data: Dict) -> int:
    """Obtener número total de frames de animación"""
    if not formatted_data:
        return 0
    first_key = list(formatted_data.keys())[0]
    return len(formatted_data[first_key])