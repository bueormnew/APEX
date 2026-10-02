# ⚡ APEX: Next-Generation Hybrid Autoregressive Architecture

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**APEX** es una librería y arquitectura híbrida de modelado secuencial y modelos de lenguaje autorregresivos (LLMs) de vanguardia. Combina de manera sinérgica cuatro bloques matemáticos de última generación:

1. **Hop-Mix**: Enrutamiento multiescala $O(\log N)$ con saltos dinámicos y ranuras de atención guiada.
2. **LRCM (Long-Range Convolutional Memory)**: Convoluciones causales jerárquicas y atención en ventanas locales para context memory.
3. **Mamba-3**: Modelo de Espacio de Estados (SSM) continuo discretizado con aproximación trapezoidal de 2º orden, rotaciones complejas (RoPE) y formulación MIMO de rango múltiple.
4. **ECHO (Nativo en reemplazo de FFN)**: Memoria asociativa de llaves producto (PKM) y micro-expertos factorizados de bajo rango con compresión compartida. **Reemplaza por completo a los FFNs tradicionales** manteniendo estabilidad superior y cero pérdida de gradientes.

---

## 🚀 Características Principales

- **Maquetas de Bloques Profesionales**:
  - `APEX_PYRAMID` (Topología ganadora: `HOP -> LRCM -> MAMBA-3 -> MAMBA-3 -> LRCM -> HOP`): Máximo equilibrio entre perplejidad, velocidad de inferencia y retención de memoria a largo plazo.
  - `APEX_OMNI` (`LRCM -> HOP -> MAMBA-3 -> MAMBA-3 -> LRCM -> MAMBA-3`): Máximo throughput computacional.
  - `APEX_DEEP_MAMBA` (`LRCM -> MAMBA-3 -> MAMBA-3 -> MAMBA-3 -> HOP -> LRCM`): Razonamiento causal intensivo.
- **Formato Unificado `.apex`**: Exportación e importación en un único archivo comprimido autocontenido que preserva pesos, configuración exacta y metadatos sin dispersión de archivos.
- **Entrenador Multitécnica (`APEXTrainer`)**: AdamW acoplado, Cosine Annealing con Warmup, Gradient Clipping por norma L2, pérdida auxiliar balanceada de ECHO y modos de fine-tuning selectivo (`echo_only`, `mamba_only`, `backbone`).
- **Suite de Métricas y Benchmarks**: Monitoreo de throughput (tokens/s), latencia por token (ms/token), uso de memoria y arnés de evaluación para pares de preguntas/respuestas (QA) y pruebas sintéticas de aguja en el pajar (*Needle-in-a-Haystack*).
- **Kernels Triton opcionales**: Scan recurrente Mamba-3 con backward explícito y actualización recurrente fused para decode de un token.

---

## 📦 Instalación

Clona el repositorio e instala en modo editable:

```bash
git clone https://github.com/bueormnew/APEX.git
cd APEX
pip install -e .
```

Requisitos: `torch>=2.0.0`, `einops>=0.7.0`.

---

## 🛠️ Guía Rápida de Uso

### 1. Crear un Modelo con un Preset

```python
from apex_core import APEXModel, BlockPreset

# Instanciar el preset ganador APEX_PYRAMID
model = APEXModel.from_preset(
    BlockPreset.APEX_PYRAMID,
    vocab_size=50257,
    d_model=256,
    max_seq_len=2048,
)

print(f"Parámetros totales: {model.count_parameters():,}")
```

### 2. Personalizar la Arquitectura e Hiperparámetros

```python
from apex_core import APEXConfig, APEXModel

custom_config = APEXConfig(
    vocab_size=32000,
    d_model=512,
    # Configurar topología arbitraria de capas
    layer_pattern=["lrcm", "mamba3", "hopmix", "mamba3", "lrcm"],
    # Hiperparámetros de ECHO (Memoria Asociativa FFN)
    echo_n_keys=64,
    echo_top_k=4,
    echo_rank=32,
    # Hiperparámetros de Mamba-3
    mamba_d_state=64,
    mamba_headdim=64,
    mamba_is_mimo=True,
)

model = APEXModel(custom_config)
```

### 3. Entrenar y Hacer Fine-Tuning

```python
from torch.utils.data import DataLoader
from apex_core import APEXTrainer, TrainingConfig

# Configurar técnicas de entrenamiento
train_cfg = TrainingConfig(
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_steps=100,
    epochs=5,
    lr_scheduler_type="cosine",
    aux_loss_weight=0.01,  # Factor de pérdida de balanceo para micro-expertos ECHO
    save_checkpoint_path="checkpoints/apex_best.apex",
)

trainer = APEXTrainer(model, train_cfg)

# (Opcional) Fine-tuning eficiente congelando capas:
# trainer.freeze_components("echo_only")   # Entrena solo la memoria asociativa ECHO
# trainer.freeze_components("mamba_only")  # Entrena solo el espacio de estados Mamba-3

# Entrenar
trainer.train(train_dataloader, val_dataloader)
```

### 4. Guardar y Cargar en Formato Único `.apex`

```python
# Exportar modelo completo a un solo archivo
model.save_apex("mi_modelo_apex.apex")

# Cargar en otra máquina o servicio de inferencia en una sola línea
loaded_model = APEXModel.load_apex("mi_modelo_apex.apex", device="cuda")

# Generar texto de forma autorregresiva
tokens = loaded_model.generate(prompt_tokens, max_new_tokens=50, temperature=0.7)
```

### 5. Medir Rendimiento y Ejecutar Benchmarks

```python
from apex_core import MetricsMonitor, BenchmarkHarness

# Medir tokens por segundo y consumo de memoria
metrics = MetricsMonitor.benchmark_throughput(model, seq_len=512, batch_size=4)
print(f"Forward: {metrics['forward_tokens_per_sec']:.1f} tok/s")
print(f"Training: {metrics['train_tokens_per_sec']:.1f} tok/s")
print(f"Memoria: {metrics['memory_mb']:.2f} MB")

# Medir velocidad de decodificación token por token
gen_metrics = MetricsMonitor.benchmark_generation(model, prompt_len=32, gen_tokens=64)
print(f"Inferencia: {gen_metrics['generation_tokens_per_sec']:.1f} tok/s ({gen_metrics['latency_ms_per_token']:.2f} ms/tok)")

# Arnés para evaluar preguntas y respuestas
harness = BenchmarkHarness(model)
qa_dataset = [
    {"question": "Capital de Francia?", "target": "Paris"},
    {"question": "2 + 2 = ", "target": "4"},
]
results = harness.evaluate_qa(qa_dataset, max_new_tokens=8, match_mode="contains")
print(f"Precisión QA: {results['accuracy_percent']:.1f}%")
```

---

## 🔬 Aceleración por Hardware (Kernels Triton)

Los kernels Triton opcionales aceleran **el núcleo recurrente de Mamba-3**, no la arquitectura híbrida completa:
- **Entrenamiento**: scan recurrente causal con backward Triton explícito. El backward guarda los estados por paso; no implementa todavía recomputación por chunks ni promete ahorros de VRAM.
- **Inferencia**: un paso de decode fusiona actualización del estado y lectura recurrente por lote/cabeza. Las proyecciones Mamba-3, Hop-Mix, LRCM y ECHO siguen ejecutándose en PyTorch.
- **Fallback**: sin Triton o CUDA, Mamba-3 usa su ruta PyTorch.

Instala el extra CUDA en Linux con `pip install -e ".[cuda]"`. Para ejecutar las pruebas y medir los kernels en una máquina Kaggle con dos T4:

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status gersonbuenahora/apex-two-t4-triton-benchmarks
```

El notebook exige dos T4, comprueba salidas y gradientes contra PyTorch, entrena el modelo híbrido con DataParallel en ambas GPU, y prueba generación en cada GPU. Las métricas se miden en el runtime y dependen de sus formas de tensores; no se asume una aceleración por adelantado. Consulta [`TRITON_KERNELS_APEX.md`](TRITON_KERNELS_APEX.md) para detalles y limitaciones.

---

## 📂 Estructura del Repositorio

```
APEX/
├── apex_core/                     # Paquete principal de la librería
│   ├── __init__.py                # Exportación de clases principales
│   ├── config.py                  # APEXConfig y BlockPreset
│   ├── model.py                   # APEXModel con serialización .apex
│   ├── trainer.py                 # APEXTrainer multi-técnica
│   └── evaluation.py             # MetricsMonitor y BenchmarkHarness
├── echo.py                        # Bloque ECHO nativo (reemplazo FFN)
├── hopmix.py                      # Bloque Hop-Mix
├── lrcm.py                        # Bloque LRCM
├── mamba3.py                      # Bloque Mamba-3 puro
├── apex_triton.py                 # Kernels Triton Mamba-3 opcionales
├── hybrid_model.py                # Ensamblado del modelo causal híbrido
├── test_apex_library.py           # Suite integral de verificación
├── tests/test_triton_kernels.py   # Pruebas CUDA de salida, gradientes y decode
├── kaggle/                        # Notebook de validación con dos T4
├── setup.py                       # Empaquetado pip
├── TRITON_KERNELS_APEX.md         # Documento detallado de Kernels en Triton
└── README.md                      # Documentación principal
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia Apache 2.0. Consulta el archivo `LICENSE` para más detalles.
