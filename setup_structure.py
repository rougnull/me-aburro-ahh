#!/usr/bin/env python3
"""
Script de instalación de la estructura modular de NeuroMechFly Renderer v2.0
Crea todos los directorios y archivos necesarios
"""

from pathlib import Path
import shutil

print("=" * 70)
print("Instalador de NeuroMechFly Enhanced Renderer v2.0")
print("=" * 70)

# Estructura de directorios
STRUCTURE = {
    "src": {
        "core": [
            "__init__.py",
            "config.py",
            "data.py",
            "model.py"
        ],
        "render": [
            "__init__.py",
            "mujoco_renderer.py"
        ]
    },
    "outputs": {
        "kinematic_replay": []
    },
    "data": {
        "inverse_kinematics": []
    }
}

def create_structure():
    """Crea la estructura de directorios"""
    print("\n[1/3] Creando estructura de directorios...")
    
    for parent, children in STRUCTURE.items():
        parent_path = Path(parent)
        parent_path.mkdir(exist_ok=True)
        print(f"  ✓ {parent}/")
        
        if isinstance(children, dict):
            for subdir, files in children.items():
                subdir_path = parent_path / subdir
                subdir_path.mkdir(exist_ok=True)
                print(f"    ✓ {parent}/{subdir}/")
        elif isinstance(children, list):
            # Es una lista de archivos
            pass

def copy_module_files():
    """Copia los archivos de módulos desde outputs"""
    print("\n[2/3] Copiando archivos de módulos...")
    
    outputs_dir = Path("outputs")
    
    # Mapeo de archivos a copiar
    files_to_copy = {
        "src_core_init.py": "src/core/__init__.py",
        "src_core_data.py": "src/core/data.py",
        "src_render_init.py": "src/render/__init__.py",
    }
    
    for source_name, dest_path in files_to_copy.items():
        source = outputs_dir / source_name
        dest = Path(dest_path)
        
        if source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, dest)
            print(f"  ✓ {dest}")
        else:
            print(f"  ⚠️  No encontrado: {source}")

def create_init_files():
    """Crea archivos __init__.py vacíos donde falten"""
    print("\n[3/3] Verificando archivos __init__.py...")
    
    for parent, children in STRUCTURE.items():
        if isinstance(children, dict):
            for subdir, files in children.items():
                init_file = Path(parent) / subdir / "__init__.py"
                if not init_file.exists() and "__init__.py" in files:
                    init_file.touch()
                    print(f"  ✓ Creado: {init_file}")

def verify_installation():
    """Verifica que la instalación esté completa"""
    print("\n" + "=" * 70)
    print("Verificando instalación...")
    print("=" * 70)
    
    required_files = [
        "src/core/__init__.py",
        "src/core/config.py",
        "src/core/data.py",
        "src/core/model.py",
        "src/render/__init__.py",
        "src/render/mujoco_renderer.py",
    ]
    
    all_exist = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ FALTA: {filepath}")
            all_exist = False
    
    if all_exist:
        print("\n✓ ¡Instalación completa!")
        print("\nPuedes ejecutar:")
        print("  python render_enhanced_3d_v2.py")
    else:
        print("\n✗ Instalación incompleta")
        print("\nArchivos faltantes detectados.")
        print("Verifica que todos los archivos .py estén en la carpeta outputs/")

def main():
    """Función principal"""
    try:
        create_structure()
        copy_module_files()
        create_init_files()
        verify_installation()
        
        print("\n" + "=" * 70)
        print("Proceso completado")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error durante la instalación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()