#!/usr/bin/env python3
"""
FlyGym Installation Diagnostic Tool
Detects the structure and available components of your FlyGym installation
"""

import sys

print("=" * 70)
print("FlyGym Installation Diagnostic")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version:")
print(f"   {sys.version}")

# Try to import flygym
print(f"\n2. Importing FlyGym...")
try:
    import flygym
    print(f"   ✓ FlyGym imported successfully")
    print(f"   Location: {flygym.__file__}")
except ImportError as e:
    print(f"   ✗ Cannot import FlyGym: {e}")
    sys.exit(1)

# Check version
print(f"\n3. FlyGym Version:")
try:
    print(f"   {flygym.__version__}")
except AttributeError:
    print(f"   ⚠️  __version__ not available")

# List all available attributes
print(f"\n4. Available attributes in 'flygym' module:")
attrs = [attr for attr in dir(flygym) if not attr.startswith('_')]
for attr in attrs[:20]:  # Show first 20
    print(f"   - {attr}")
if len(attrs) > 20:
    print(f"   ... and {len(attrs) - 20} more")

# Try different import patterns
print(f"\n5. Testing import patterns:")

patterns = [
    ("flygym.Fly", "from flygym import Fly"),
    ("flygym.fly.Fly", "from flygym.fly import Fly"),
    ("flygym.core.Fly", "from flygym.core import Fly"),
    ("flygym.simulation.Fly", "from flygym.simulation import Fly"),
    ("flygym.arena.Fly", "from flygym.arena import Fly"),
    ("flygym.Camera", "from flygym import Camera"),
    ("flygym.camera.Camera", "from flygym.camera import Camera"),
    ("flygym.SingleFlySimulation", "from flygym import SingleFlySimulation"),
    ("flygym.simulation.SingleFlySimulation", "from flygym.simulation import SingleFlySimulation"),
]

successful_imports = []

for module_path, import_statement in patterns:
    try:
        exec(import_statement)
        print(f"   ✓ {import_statement}")
        successful_imports.append(import_statement)
    except (ImportError, AttributeError) as e:
        print(f"   ✗ {import_statement}")

# Check for submodules
print(f"\n6. Available submodules:")
import pkgutil
import importlib

def find_submodules(package, prefix=""):
    try:
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, prefix=package.__name__ + "."):
            print(f"   - {modname}")
            if ispkg and len(modname.split('.')) <= 2:  # Only go 2 levels deep
                try:
                    submod = importlib.import_module(modname)
                    find_submodules(submod, modname + ".")
                except:
                    pass
    except AttributeError:
        pass

find_submodules(flygym)

# Check for specific classes we need
print(f"\n7. Checking for required classes:")

required_classes = [
    ('Fly', ['flygym', 'flygym.fly', 'flygym.core', 'flygym.simulation', 'flygym.arena']),
    ('Camera', ['flygym', 'flygym.camera', 'flygym.core']),
    ('SingleFlySimulation', ['flygym', 'flygym.simulation', 'flygym.core']),
]

found_classes = {}

for class_name, possible_modules in required_classes:
    print(f"\n   Searching for '{class_name}':")
    for module_path in possible_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                print(f"   ✓ Found in {module_path}.{class_name}")
                found_classes[class_name] = f"{module_path}.{class_name}"
                break
        except:
            pass
    else:
        print(f"   ✗ Not found in any expected location")

# Summary
print(f"\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if successful_imports:
    print(f"\n✓ Working imports:")
    for imp in successful_imports:
        print(f"  {imp}")
else:
    print(f"\n✗ No successful imports found!")

if found_classes:
    print(f"\n✓ Found required classes:")
    for class_name, location in found_classes.items():
        print(f"  {class_name}: {location}")
else:
    print(f"\n✗ No required classes found!")

print(f"\n" + "=" * 70)
print("Please share this output to get a compatible script!")
print("=" * 70)