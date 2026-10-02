# APEX: kernels Triton disponibles y validacion

## Alcance implementado

`apex_triton.py` acelera el nucleo recurrente usado por el bloque Mamba-3 de APEX. No es una fusion de extremo a extremo de Mamba-3, HOP-MIX, LRCM y ECHO: sus proyecciones, RoPE, puertas, bloques de memoria y ECHO siguen ejecutandose en PyTorch. La seleccion automatica requiere CUDA y Triton; sin ellos, Mamba-3 usa la ruta PyTorch existente.

### Entrenamiento: scan recurrente y backward

`mamba3_scan(u, b, c, dt, trap, a)` procesa tensores contiguos con formas:

- `u`: `[B, T, H, P]`
- `b`, `c`: `[B, T, H, N]`
- `dt`, `trap`: `[B, T, H]`
- `a`: `[H]`, decaimiento continuo negativo
- salida: `[B, T, H, P]`

Una instancia Triton por `(batch, head)` itera causalmente por `T`, actualiza la memoria trapezoidal `bx_prev`, el estado SSM y la lectura `C`. Un `torch.autograd.Function` asociado calcula gradientes para las seis entradas. La implementacion guarda `h` y `B*x` por paso para el backward; **no** es un scan paralelo por chunks, no recomputa activaciones y no garantiza una reduccion de memoria. Las dimensiones de estado y ancho se rellenan internamente a la siguiente potencia de dos.

La precision de acumulacion interna es FP32; la salida/activaciones guardadas tienen el dtype de `u`. El alcance probado en el notebook es FP32. No se debe inferir soporte numerico FP16/BF16/FP8 de esa prueba.

### Inferencia: actualizacion y lectura de un paso

`mamba3_decode_step(...)` fusiona el calculo de `B*x`, la mezcla trapezoidal, la actualizacion del estado y la reduccion con `C` para un token. Hay una instancia por `(batch, head)`; `h_state` y `bx_prev` se actualizan in-place y la funcion devuelve `None` si no hay CUDA/Triton, para que `Mamba3.step` use su ruta PyTorch. Las proyecciones del bloque, RoPE, ECHO, HOP-MIX y LRCM **no** estan fusionadas en este kernel.

## Instalar, probar y medir

En Linux con GPU NVIDIA, instala APEX con el extra:

```bash
pip install -e ".[cuda]"
```

En el repositorio se incluye `kaggle/apex_two_t4_triton.ipynb`, configurado como kernel Kaggle privado con `machine_shape: NvidiaTeslaT4` (T4 x2), GPU e internet habilitados. Se publica y ejecuta con:

```bash
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4
kaggle kernels status gersonbuenahora/apex-two-t4-triton-benchmarks
kaggle kernels logs gersonbuenahora/apex-two-t4-triton-benchmarks
kaggle kernels output gersonbuenahora/apex-two-t4-triton-benchmarks -p kaggle-output
```

El notebook confirma cantidad/modelo de GPU y version de Triton, compara forward y backward del scan con una referencia PyTorch, compara el decode y la mutacion de estado, ejecuta un paso de entrenamiento del modelo HOP-MIX + LRCM + Mamba-3 con `DataParallel` en las dos GPU, genera tokens en ambas y mide el scan de entrenamiento y el paso de decode por GPU. Tambien corre `apex train` en FP16 con `DataParallel` sobre ambas T4, valida, exporta el modelo y checkpoint reanudable, y ejecuta `apex generate`. Produce `apex_triton_benchmark.json` con las formas, tiempos y cocientes medidos; no contiene cifras supuestas. Los resultados dependen del runtime y de las formas probadas y no equivalen a un benchmark de un LLM grande.

### Resultado guardado de Kaggle

La ejecucion completada del 2026-10-02 (notebook version 7, commit `1568442`, PyTorch 2.11.0+cu128, Triton 3.6.0) produjo [`kaggle/results/apex_triton_benchmark.json`](kaggle/results/apex_triton_benchmark.json):

| Medicion | PyTorch | Triton | Cociente |
|---|---:|---:|---:|
| Scan de entrenamiento forward + backward, `(B,T,H,P,N)=(4,96,4,32,32)` | 103.864 ms | 1.052 ms | 98.731x |
| Decode, batch 16, GPU 0 | 0.2509 ms | 0.0704 ms | 3.565x |
| Decode, batch 16, GPU 1 | 0.2642 ms | 0.0732 ms | 3.609x |

Son microbenchmarks del nucleo recurrente frente a la implementacion PyTorch de referencia; **no** son speedups de entrenamiento o generacion del modelo completo. El mismo run verifico cuatro pruebas Triton (forward/backward y decode en las dos T4), una actualizacion de entrenamiento hibrido DataParallel con logits `[8,32,128]`, memoria reservada en ambas tarjetas y generacion de forma `[1,7]` en cada una.

El CLI integrado completo tambien termino dos pasos de entrenamiento FP16 sobre ambas T4, midio validacion, guardo `latest.apex` y `training_state.pt`, y genero texto desde el artefacto exportado. Es una prueba de integracion/smoke test con un corpus diminuto, no una medida de calidad del modelo ni un entrenamiento a escala de produccion.

Prueba local de la suite:

```bash
python -m pytest -q
```

En una maquina CPU-only las pruebas Triton CUDA se marcan como omitidas. El notebook de Kaggle es el test requerido para validar compilacion, numerica, rendimiento y uso real de las dos T4. No declarar aceleracion, precision mixta, ahorro de memoria o estado de produccion para otras GPU/formas sin repetir esas pruebas en el hardware objetivo.
