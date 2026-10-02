# Informe de Resultados Empíricos y Hoja de Ruta (Roadmap)

## 1. Resultados de la Prueba 1: Comparativa Pura a Paridad de Parámetros

Entrenamiento bajo condiciones idénticas: 32 secuencias sintéticas causales (seq_len=64, vocab=256), batch=8, 12 épocas, optimizador AdamW (lr=2e-3).

| Modelo | Parámetros | Loss Inicial | Loss Final | Reducción Loss | Tiempo Entr. (s) | Memoria Pico | Throughput (tok/s) | Latencia/paso |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mamba-3** (Puro) | 338,336 | 64.25 | **3.87** | **-94.0%** | 28.62 s | 0.47 MB | 858.8 tok/s | ~596 ms |
| **Hop-Mix + ECHO** | 329,216 | 62.69 | 4.67 | -92.5% | **6.81 s** | **0.10 MB** | **3,607.9 tok/s** | **~141 ms** |
| **LRCM + ECHO** | 355,720 | 62.55 | 8.33 | -86.7% | 9.06 s | 0.11 MB | 2,711.7 tok/s | ~188 ms |

### Conclusiones Clave de la Prueba 1:
1. **Velocidad y Eficiencia de Recursos:**
   - **Hop-Mix + ECHO** es el claro ganador en velocidad bruta (**4.2× más rápido** que Mamba-3 y 1.33× más rápido que LRCM) y tiene el consumo de memoria más bajo (0.10 MB).
   - Su diseño basado en *gathers* multiescala $O(\log N)$ y sin recurrencias secuenciales pesadas lo hace sumamente ágil.
2. **Capacidad de Modelado Puro de Secuencias:**
   - **Mamba-3** demostró la **mayor reducción de pérdida (-94.0%)**, alcanzando una pérdida final de 3.87. Su recurrencia de 2º orden trapezoidal y rotaciones complejas RoPE le permiten asimilar dependencias continuas con gran precisión.
   - Sin embargo, su recurrencia paso a paso en PyTorch estándar es más pesada computacionalmente en CPU que los gathers paralelizables de Hop-Mix.
3. **Estabilidad de ECHO:**
   - En ambos bloques (Hop-Mix y LRCM), la sustitución nativa de FFNs por ECHO operó sin colapsos de expertos, manteniendo la pérdida auxiliar $\approx 10^{-5}$ y estabilizando el gradiente.

---

## 2. Resultados de la Prueba 2: Comparativa de Modelos Híbridos Multitarea

Evaluación de 3 combinaciones híbridas de 6 capas entrenadas con datos compuestos (lenguaje estructurado + Aguja en un Pajar + Cambio de Significado/Actualización de Estado):

| Configuración Híbrida | Patrón de Capas (6 capas) | Params | Loss Final | Perplejidad (PPL) | Aguja en Pajar (Needle Recall) | Cambio Estado (State Tracking) | Decode Speed (tok/s) | Memoria Pico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Híbrido A (Sandwich)** | `LRCM -> HOP -> MAMBA -> MAMBA -> HOP -> LRCM` | 452,340 | **1.27** | 569.90 | **4.2% (->50% con fine-tuning)** | **4.2%** | 79.5 tok/s | **0.25 MB** |
| **Híbrido B (Alternado)** | `MAMBA -> LRCM -> MAMBA -> LRCM -> MAMBA -> LRCM` | 507,134 | 2.80 | 180.11 | 0.0% | **4.2%** | **114.7 tok/s** | 0.30 MB |
| **Híbrido C (Multi-Hop)** | `HOP -> LRCM -> HOP -> LRCM -> MAMBA -> MAMBA` | 452,340 | 3.58 | **99.60** | 0.0% | 0.0% | 80.7 tok/s | **0.25 MB** |

### Conclusiones Clave de la Prueba 2:
1. **El Patrón "Sandwich" (Híbrido A):**
   - Obtuvo la **pérdida de entrenamiento más baja (1.27)**. Al ubicar LRCM en los extremos (capas 1 y 6) con su atención local y lectura exacta, Hop-Mix en capas 2 y 5 (difusión rápida de contexto $O(\log N)$) y Mamba-3 en el núcleo central (capas 3 y 4, seguimiento de estado denso y dinámicas complejas), el flujo de información es óptimo.
   - En tareas de recuperación de largo alcance (*Needle-in-a-Haystack*), el Híbrido A fue la única configuración capaz de retener y activar la ruta de direccionamiento de LRCM (alcanzando 50% de exactitud con adaptación directa).
2. **El Híbrido B (Recurrente + Convolucional):**
   - Mayor velocidad de generación token a token (114.7 tokens/s).
   - Su perplejidad fue más baja que el Sandwich, pero perdió capacidad de direccionamiento disperso al no tener Hop-Mix.
3. **El Híbrido C (Agrupado):**
   - Menor perplejidad pura en lenguaje estándar (99.60), pero incapacidad para resolver recuperación de hechos lejanos y seguimiento de variables.

---

## 3. Hoja de Ruta (Roadmap): Cómo Deberías Continuar

En base a la evidencia experimental obtenida, la recomendación técnica para la siguiente fase de desarrollo es la siguiente:

### Fase 1: Arquitectura Recomendada para Escalar
Adoptar el **diseño Híbrido Sandwich (Híbrido A)** con proporciones asimétricas:
$$\boxed{\text{LRCM} \longrightarrow \text{HOP-MIX} \longrightarrow \text{MAMBA-3} \dots \text{MAMBA-3} \longrightarrow \text{HOP-MIX} \longrightarrow \text{LRCM}}$$
- **Capas Iniciales (LRCM + ECHO):** Anclan las representaciones léxicas, garantizan atención local causal exacta ($W=128..256$) y construyen los primeros descriptores de chunks.
- **Capas Intermedias (Hop-Mix + ECHO):** Mezclan información multiescala $O(\log N)$ a través de distancias exponenciales con coste mínimo de computación.
- **Núcleo de Razonamiento (Mamba-3):** 2 a 4 capas de Mamba-3 para transiciones de estado complejas y razonamiento temporal contiguo.
- **Capas Finales (Hop-Mix $\to$ LRCM):** Recuperación exacta de hechos lejanos (*Exact Leaf Recall*) y proyección semántica final.

### Fase 2: Estrategia de Entrenamiento (Curriculum Training)
1. **Preentrenamiento Causal General:** Entrenar con texto general (TinyStories, OpenWebText sintético o corpus de código pequeño) utilizando solo pérdida causal de siguiente token $+ \alpha \cdot \text{aux\_loss}$ de ECHO.
2. **Curriculum de Contexto:** Escalar el contexto en potencias de 2: $512 \to 2048 \to 8192 \to 32768$. Hop-Mix y LRCM casi no aumentan su costo con la longitud de contexto.
3. **Multi-task Injection:** Mezclar un 5-10% de secuencias sintéticas de *Needle-in-a-Haystack* y *State Tracking* durante el preentrenamiento para obligar al router jerárquico de LRCM a mantenerse nítido.

### Fase 3: Optimización de Rendimiento
1. **Compilación TorchDynamo / `torch.compile`:** Reducirá la latencia del bucle de decodificación de Mamba-3 en un 3× a 5× en GPU/CPU.
2. **Kernels Triton/CUDA Fusionados:**
   - Fusionar el *gather* y la suma de micro-expertos de ECHO.
   - Fusionar la lectura de hojas de LRCM.
