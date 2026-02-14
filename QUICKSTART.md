# NeuroMechFly 3D Embodied Simulation - Quick Start

Complete neural-driven fly simulation with realistic 3D kinematics and closed-loop sensorimotor control.

## 30-Second Setup & Run

```bash
# 1. Activate environment  
.\.venv\Scripts\activate

# 2. Run quick demo
python demo_embodied.py --duration 10

# 3. Watch output (completes in ~4 seconds):
#    - Real-time statistics
#    - Neural spike counts
#    - Behavioral metrics
#    - Position tracking
```

## What's Included

✅ **Neural Circuit**: 50 ORN → 2000 KC → 10 DN (spiking neurons)
✅ **3D Body**: Realistic skeleton with 6 legs (18 DOF)
✅ **Motor Control**: CPG-driven tripod walking
✅ **Environment**: 100×100×50 mm arena with Gaussian odor
✅ **Simulation**: Complete embodied cognition loop

## Quick Commands

```bash
# Demo (10 seconds, fast)
python demo_embodied.py --duration 10

# Full simulation (30 seconds with visualization)
python run_3d_simulation.py --duration 30

# Quick test (1 second, verify setup)
python demo_embodied.py --duration 1

# Long run (60 seconds, full behavior)
python run_3d_simulation.py --duration 60
```

## Understanding the Output

```
Step:    8000 | Pos: (  3.02,   2.01, 0.00) mm | Velocity:   0.0050 mm/s | Odor: 0.018
```

- **Step**: Timestep number (1000 = 1 second virtual time)
- **Pos**: Fly position (x, y, z in millimeters)
- **Velocity**: Movement speed (mm/s)
- **Odor**: Detected odor concentration (0-1)

**Final Statistics Include**:
- Total distance traveled
- Mean/max velocity
- Neural spike counts
- Behavioral metrics

## Key Architecture

```
[Odor Input] → [Neural Brain] → [Motor Command] → [Leg Movement] 
                                                         ↓
                                            [Updated Position] 
                                                         ↓
                                        [New Odor Detected]
```

The loop closes automatically! Neural activity drives behavior which changes sensory input.

## Configuration

Quick parameter changes in YAML files:

`config/environment.yaml`:
```yaml
arena:
  width: 100       # mm
  height: 100      # mm
food_position: [50, 50, 0]

odor:
  food_intensity: 1.0      # Strong odor source
  diffusion_coefficient: 0.1
```

`config/brain_params.yaml`:
```yaml
n_orn: 50          # Olfactory neurons
n_kc: 2000         # Kenyon cells
n_dn: 10           # Motor output neurons
```

`config/fly_params.yaml`:
```yaml
motor_gains:
  forward_speed: 20.0     # mm/s per DN
  rotation_speed: 45.0    # degrees/s per DN
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `.\.venv\Scripts\activate` first |
| No spikes recorded | Normal if far from odor. Try longer: `--duration 60` |
| Fly moves slowly | Realistic! CPG output is small but continuous |
| Vispy window won't open | Use `demo_embodied.py` instead for console output |

## Example: Running Different Experiments

### Exp 1: Verify Setup (30 seconds total)
```bash
python demo_embodied.py --duration 1
# ✓ Should complete almost instantly
# ✓ Shows it's working
```

### Exp 2: Normal Behavior (4-5 seconds total)
```bash
python demo_embodied.py --duration 10
# ✓ Full statistics
# ✓ Neural activity
# ✓ Trajectory data
```

### Exp 3: Extended Behavior (60-70 seconds total)
```bash
python run_3d_simulation.py --duration 60
# ✓ Comprehensive data
# ✓ HDF5 export
# ✓ 7 visualization plots
# ✓ Output in data/20260214_*/
```

## Project Structure

```
NeuroMechFly Sim/
├── config/
│   ├── environment.yaml    ← Arena parameters
│   ├── brain_params.yaml   ← Neural circuit
│   └── fly_params.yaml     ← Motor control
├── core/
│   ├── simulation.py       ← Main loop
│   └── environment.py      ← Physics
├── brain/
│   └── olfactory_circuit.py ← Spiking neurons
├── body/
│   └── realistic_body.py   ← 3D skeleton + kinematics
├── analysis/
│   ├── visualization.py    ← 2D plots
│   └── visualization_3d.py ← 3D plots
└── [Demo scripts]
    ├── demo_embodied.py    ← Use this! 🎯
    ├── run_3d_simulation.py
    └── run_experiment.py
```

## Next Steps

1. **Run demo**: `python demo_embodied.py --duration 10`
2. **Read docs**: See README.md for full documentation
3. **Modify**: Edit config files to change behavior
4. **Experiment**: Try different durations and parameters
5. **Extend**: Add features like learning, vision, or new behaviors

## Key Concepts

- **Embodied Cognition**: Brain + body + environment form closed loop
- **Biophysical Realism**: Based on Drosophila connectomics data
- **Spiking Neurons**: LIF model with realistic dynamics
- **3D Kinematics**: Realistic leg movements from motor commands
- **Closed-Loop**: Sensory feedback continuously influences behavior

---

## Start Now!

```bash
python demo_embodied.py --duration 10
```

This will:
1. Initialize neural circuit, fly body, and arena (~50 ms)
2. Run 10 seconds of simulation (10,000 timesteps at 1 ms each)
3. Print statistics showing behavior and neural activity
4. Complete in about 4 seconds wall-clock time

Enjoy exploring embodied cognition! 🧠🦗

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

