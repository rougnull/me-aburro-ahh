# NeuroMechFly - Quick Start Guide

## ¿Qué es Este Proyecto?

Un **simulador embodied** que integra un modelo neuronal olfativo con un cuerpo físico simulado. Tu cerebro virtual controla una mosca que debe navegar hacia una fuente de olor en una arena virtual.

---

## 1️⃣ Activación Rápida

### Windows (PowerShell):
```powershell
cd "C:\Users\eduar\Documents\Workspace\NeuroMechFly Sim\proyecto_mosca"
..\venv\Scripts\Activate.ps1
```

### Windows (CMD):
```cmd
cd C:\Users\eduar\Documents\Workspace\NeuroMechFly Sim\proyecto_mosca
..\venv\Scripts\activate.bat
```

### macOS/Linux:
```bash
cd ~/NeuroMechFly\ Sim/proyecto_mosca
source ../venv/bin/activate
```

---

## 2️⃣ Prueba Rápida (Demo)

```bash
python demo.py
```

**Salida esperada:**
```
✓ Arena initialized (100x100x50 mm)
✓ Brain initialized (10 descending neurons)
✓ Fly body initialized (mock mode)

Simulating...
Step 1000/5000 | Position: (50.0, 50.0) | Odor: 1.000
...
✓ Demo complete!
```

---

## 3️⃣ Ejecutar Simulación Completa

```bash
# Simulación de 60 segundos
python run_experiment.py --duration 60
```

**Salida:**
- Datos guardados en `data/YYYYMMDD_HHMMSS/simulation_data.h5`
- Gráficos generados:
  - `trajectory.png` - Trayectoria de la mosca
  - `neural_activity.png` - Actividad de neuronas descendentes
  - `odor_response.png` - Detección de olor

---

## 4️⃣ Modificar Configuración

Edita los archivos YAML en `config/`:

### `config/environment.yaml` - Arena y Olor
```yaml
arena:
  width: 100.0              # Amplitud (mm)
  height: 100.0             # Profundidad (mm)

odor:
  food_position: [50.0, 50.0, 0.0]  # Posición de comida
  food_intensity: 1.0
```

### `config/brain_params.yaml` - Neurones
```yaml
neurons:
  orn_count: 50             # Receptores olfatorios
  kc_count: 2000            # Cuerpo maduro
  dn_count: 10              # Neuronas descendentes
```

### `config/fly_params.yaml` - Motor
```yaml
motor_gains:
  forward_speed: 20.0       # mm/s por unidad de comando
  rotation_speed: 45.0      # deg/s por unidad de comando
```

---

## 5️⃣ Estructura de Directorios

```
proyecto_mosca/
├── config/                 # Configuraciones YAML (EDITABLE)
├── core/                   # Lógica principal
│   ├── simulation.py       # Loop principal
│   └── environment.py      # Arena y olor
├── brain/                  # Red neuronal olfativa
│   ├── olfactory_circuit.py
│   ├── sensory_transduction.py
│   └── descending_interface.py
├── body/                   # Interfaz del cuerpo
│   └── fly_interface.py
├── data/                   # SALIDA (resultados)
├── demo.py                 # Demo 5 segundos
└── run_experiment.py       # Script principal
```

---

## 6️⃣ Archivos Generados

Después de ejecutar `run_experiment.py`, encuentra los resultados en:

```
data/20260214_224413/
├── simulation_data.h5     # Datos crudos (HDF5)
├── trajectory.png         # Gráfico de trayectoria
├── neural_activity.png    # Spikes de neuronas
└── odor_response.png      # Detección olfatoria
```

### Leer datos HDF5 en Python:
```python
import h5py
import numpy as np

with h5py.File('data/20260214_224413/simulation_data.h5', 'r') as f:
    position = np.array(f['position'])  # Shape: (steps, 3)
    odor = np.array(f['odor_input'])    # Shape: (steps,)
    print(f"Simulación: {len(position)} pasos")
    print(f"Posición final: {position[-1]}")
```

---

## 7️⃣ Ejemplos de Uso Avanzado

### Cambiar duración y config:
```bash
python run_experiment.py --duration 120 --config config/environment.yaml
```

### Ejecutar desde notebook (Jupyter):
```python
import sys
sys.path.insert(0, '.')
from core.simulation import NeuroMechFlySimulation

# ... cargar todo como en run_experiment.py
sim.run(num_steps=10000, verbose=True)
```

---

## 8️⃣ Troubleshooting

### "No module named 'core'"
```bash
# Asegúrate de ejecutar desde proyecto_mosca/
cd proyecto_mosca
python run_experiment.py
```

### "ModuleNotFoundError: No module named 'yaml'"
```bash
# Reinstalar dependencias
pip install -r requirements.txt
```

### "Permission denied" (en macOS/Linux)
```bash
chmod +x demo.py run_experiment.py
```

---

## 9️⃣ Componentes Principales

### `OlfactoryCircuit` (Brain)
- 50 ORNs → procesar olor
- 20 PNs → filtrado
- 2000 KCs → aprendizaje asociativo
- 34 MBONs → codificación de valencia
- 10 DNs → comandos motores

### `Arena` (Entorno)
- Gradiente gaussiano de olor (comida en el centro)
- Búsqueda: navegar hacia concentración máxima

### `FlyInterface` (Cuerpo)
- CPG (central pattern generator) para caminar
- Cinemática forward: DN → velocidad + rotación

---

## 🔟 Próximos Pasos

1. **Integrar NeuroMechFly real**: Reemplazar `FlyInterface` con simulador físico
2. **Implementar aprendizaje**: STDP en conexiones KC-MBON
3. **Agregar controlador RL**: Para entrenar comportamiento
4. **Validación biológica**: Comparar con experimentos reales

---

## 📚 Referencias

- [NeuroMechFly GitHub](https://github.com/NeLy-EPFL/NeuroMechFly)
- [Fruit fly connectome (eLife)](https://elifesciences.org/articles/57443)
- [Learning in Drosophila](https://www.nature.com/articles/s41593-019-0505-2)

---

## ✉️ Preguntas?

Ver `README.md` para documentación completa.

