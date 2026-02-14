# NeuroMechFly Simulation Project README

## Descripción General

Este proyecto implementa una simulación **embodied** de la mosca (Drosophila) usando **NeuroMechFly**. El objetivo es integrar un modelo neuronal olfativo con un cuerpo físico simulado en MuJoCo, permitiendo que el "cerebro virtual" controle el movimiento del cuerpo en un entorno con gradientes de olor.

## Estructura del Proyecto

```
proyecto_mosca/
├── config/                 # Configuraciones YAML
│   ├── environment.yaml    # Parámetros de la arena y olor
│   ├── fly_params.yaml     # Parámetros físicos de la mosca
│   └── brain_params.yaml   # Parámetros del modelo neuronal
│
├── core/                   # Módulo central
│   ├── simulation.py       # Bucle principal de simulación
│   └── environment.py      # Definición de la arena y física
│
├── brain/                  # Módulo neural
│   ├── olfactory_circuit.py        # Red neuronal olfativa
│   ├── sensory_transduction.py     # Conversión señal -> entrada neuronal
│   └── descending_interface.py     # Decodificación comandos motores
│
├── body/                   # Módulo del cuerpo
│   └── fly_interface.py    # Interfaz con NeuroMechFly + controlador de caminata
│
├── data/                   # Salida de simulaciones (HDF5, plots)
├── analysis/               # Scripts de análisis
│   └── visualization.py    # Funciones para graficar resultados
│
├── run_experiment.py       # Punto de entrada
└── requirements.txt        # Dependencias Python
```

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
