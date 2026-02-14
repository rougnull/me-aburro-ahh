# NeuroMechFly: Differentiable Embodied Neuroscience

Complete neural-driven fly simulation with **end-to-end learning via backpropagation through time (BPTT)**.

Integrates:
- Real FlyWire connectome data
- Differentiable spiking neural networks (PyTorch)
- Embodied physics simulation (NeuroMechFly interface)
- Task-based optimization (navigation, energy efficiency)

## What is NeuroMechFly DMN?

**DMN = Differentiable Mechanical Networks**

This framework transforms fixed-parameter simulation into **learning systems** that optimize neural circuits for behavioral tasks:

```
NeuroMechFly Physics Engine ←→ Embodied Environment
              ↑                        ↑
              │                        │
         Motor Commands ←→ Differentiable Neural Circuit
              ↑                        ↑
              └────────────────────────┘
                   Backpropagation Through Time
                        (Auto-differentiation)
                              ↓
                    Learn Synaptic Weights
                    Minimize: Distance to Goal
                             + Energy Cost
                             + Sparsity Loss
```

**Key Innovation**: Gradient descent discovers what neural mechanisms are necessary to navigate, solving the **inverse problem** of neurobiology.

## Features

### Neural Substrate
- **50 Olfactory Receptor Neurons (ORN)**: Gaussian tuning curves sensing odor gradient
- **2000 Kenyon Cells (KC)**: Sparse coding with random connectivity  
- **34 Mushroom Body Output Neurons (MBON)**: Linear summation of KC activity
- **10 Descending Neurons (DN)**: Motor command output
- **Biophysic LIF Model**: Leaky integrate-and-fire with realistic reset dynamics

### Body Model
- **3D Skeleton**: Head, thorax, abdomen
- **6 Legs**: 3 segments each with forward kinematics
- **2 Wings**: Visualization only
- **Tripod Gait CPG**: Central Pattern Generator at 10 Hz
- **Ground Contact Physics**: Foot contact detection and friction

### Environment
- **100×100×50 mm Arena**: Bounded 3D space
- **Gaussian Odor Gradient**: Centered at [50, 50, 0] mm
- **Diffusion Simulation**: Exponential falloff with distance

## Installation & Setup

### Step 1: PyTorch (2 min)
```bash
# CPU version (no GPU needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or GPU version for 10x speedup
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Dependencies (1 min)
```bash
pip install numpy scipy scikit-learn networkx h5py pyyaml matplotlib
```

### Step 3: Verify Setup (2 min)
```bash
python dmn_verify_setup.py
```

## Quick Start - Train Embodied DMN

### Basic Training (5 min on CPU, 30 sec on GPU)
```bash
# Train for 20 episodes (default)
python train_embodied_dmn.py --num-episodes 20

# Shows real-time loss, metrics, and results
```

Expected output:
```
Episode  1/20 | Loss:   12.3456 | Nav:    10.2341 | Energy:     0.0145
Episode  5/20 | Loss:    8.1234 | Nav:     7.0234 | Energy:     0.0089
Episode 20/20 | Loss:    2.1234 | Nav:     1.9876 | Energy:     0.0034

Final Trajectory Metrics:
  final_distance_to_goal: 2.3400
  mean_distance_to_goal: 15.4200
  total_distance_traveled: 45.2300

✓ Results saved to embodied_training_results/
```

### Advanced Training with Custom Parameters
```bash
# Emphasize energy efficiency
python train_embodied_dmn.py \
  --num-episodes 50 \
  --learning-rate 0.001 \
  --nav-weight 1.0 \
  --energy-weight 0.5

# Or focus on pure navigation
python train_embodied_dmn.py \
  --num-episodes 100 \
  --learning-rate 0.0001 \
  --nav-weight 2.0 \
  --energy-weight 0.01
```

## What Gets Learned?

The training process learns:

1. **Synaptic Weights** (ORN→PN, PN→KC, KC→MBON, MBON→DN)
   - Which connections matter for navigation
   - How strongly to weight inputs

2. **Neuron Time Constants** (τ)
   - How fast neurons integrate input
   - Balance between fast reactions and filtering

3. **Motor Encoding** (DN→velocity mapping)
   - What DN spike patterns produce effective movement
   - Emergent motor strategies

**All optimized via gradient descent to minimize task loss:**
- Distance to goal
- Energy expenditure
- Sparse representation maintenance

## Project Structure

## Conceptos Clave

### 1. **Capas del Control Motor**

- **Cerebro Olfativo (Brain Layer)**: Integra inputs olfatorios y genera comandos de alto nivel (avanzar, girar)
- **Interface de Neuronas Descendentes (DN Interface)**: Decodifica la actividad neuronal en comandos motores
- **Controlador de Caminata (CPG - Central Pattern Generator)**: Genera ritmos de las patas modulados por los comandos DN
- **Física (MuJoCo via NeuroMechFly)**: Simula el movimiento real con inercia, fricción y colisiones

### 2. **Circuito Olfatorio Simulado**

```
Odor (Arena) → ORNs (sensores) 
            ↓
            PNs (procesamiento primario)
            ↓
            KCs (aprendizaje asociativo - Mushroom Body)
            ↓
            MBONs (codificación de valencia)
            ↓
            DNs (comandos motores)
```

### 3. **Hipótesis de Embodied Cognition**

Cuando conectas un cerebro virtual con un cuerpo físico simulado, puedes probar si tu modelo neuronal puede genuinamente **controlar el comportamiento** considerando:
- Inercia del cuerpo
- Fricción con el terreno
- Respuesta no lineal de los actuadores
- Retroalimentación sensorial de propioceptores

## Instalación

### 1. Activar el entorno virtual

**En Windows (PowerShell):**
```powershell
cd "C:\Users\eduar\Documents\Workspace\NeuroMechFly Sim\proyecto_mosca"
..\venv\Scripts\Activate.ps1
```

**En Windows (CMD):**
```cmd
cd C:\Users\eduar\Documents\Workspace\NeuroMechFly Sim\proyecto_mosca
..\venv\Scripts\activate.bat
```

**En macOS/Linux:**
```bash
cd ~/NeuroMechFly Sim/proyecto_mosca
source ../venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. (Futuro) Instalar NeuroMechFly

Una vez disponible:
```bash
pip install neuromechfly
```

O desde fuente:
```bash
git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
pip install -e NeuroMechFly/
```

## Uso

### Ejecutar simulación básica

```bash
python run_experiment.py --duration 60 --config config/environment.yaml
```

### Parámetros disponibles

- `--duration`: Duración de la simulación en segundos (default: 60)
- `--config`: Ruta al archivo de configuración principal (default: config/environment.yaml)
- `--verbose`: Mostrar más información (optional)

### Salida

La simulación genera:
- `data/YYYYMMDD_HHMMSS/simulation_data.h5`: Datos crudos en formato HDF5
  - Trayectoria (x, y, z)
  - Spikes neuronales de cada capa
  - Inputs sensoriales (odor)
- `data/YYYYMMDD_HHMMSS/trajectory.png`: Gráfico de la trayectoria en la arena
- `data/YYYYMMDD_HHMMSS/neural_activity.png`: Actividad de neuronas descendentes
- `data/YYYYMMDD_HHMMSS/odor_response.png`: Concentración de olor detectada

## Próximos Pasos

### Fase 2: Integración con NeuroMechFly Real

1. **Instalar NeuroMechFly**: Obtener y instalar la librería completa
2. **Reemplazar mock en `body/fly_interface.py`**: Conectar con el simulador de física real
3. **Calibrar ganancias motoras**: Ajustar factores de conversión entre actividad neural y comandos motores

### Fase 3: Implementar Aprendizaje

1. **STDP (Spike-Timing-Dependent Plasticity)**: En conexiones KC-MBON
2. **Refuerzo**: Recompensa cuando la mosca llega a la comida
3. **Extinción**: Pérdida de aprendizaje sin recompensa

### Fase 4: Validación Biológica

1. Comparar patrones de movimiento con datos experimentales
2. Validar rangos de frecuencia de disparo neuronal
3. Reproducir comportamientos conocidos (quimiotaxis, trail-following, etc.)

## Referencias Biológicas

- [NeuroMechFly: Flying Fruit Fly Embodied in Physics](https://www.biorxiv.org/content/10.1101/2022.09.21.508678v1)
- [A connectome of the adult fruit fly brain](https://elifesciences.org/articles/57443) (Connectome completo)
- [Neural circuits for sensory-motor behaviors](https://www.nature.com/articles/s41593-019-0505-2)

## Notas Técnicas

### Sincronización de Timesteps

Es crítico sincronizar los timesteps de:
- **MuJoCo (physics)**: ~1 ms (0.001s) típicamente
- **Brian2 (neural)** si usas: Debe ser divisor de MuJoCo
- **Rendering**: Independiente, puede ser menos frecuente

En `config/brain_params.yaml`:
```yaml
integration:
  dt: 0.0001  # 10x más rápido que física → 10 pasos neurales por paso de física
```

### Formato de Salida HDF5

Después de una simulación, puedes cargar los datos así:

```python
import h5py
import numpy as np

with h5py.File('data/20240214_120000/simulation_data.h5', 'r') as f:
    positions = np.array(f['position'])      # Shape: (steps, 3)
    dn_spikes = np.array(f['neural_spikes'])  # Variable
    odor_input = np.array(f['odor_input'])   # Shape: (steps,)
```

## Licencia

[Especificar después]

## Contacto

Envía preguntas o sugerencias a [tu email]
