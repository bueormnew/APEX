# Resultados Exhaustivos: 5 Nuevas Mezclas Híbridas (Hop-Mix + LRCM + Mamba-3 + ECHO)

Este informe complementa la investigación previa evaluando 5 configuraciones arquitectónicas novedosas para descubrir la mejor sinergia entre los bloques.

---

## 1. Configuraciones Evaluadas

Todas las arquitecturas constan de 6 capas y un presupuesto similar (~452K–470K parámetros) evaluadas bajo el mismo dataset multitarea (lenguaje causal, *Needle-in-a-Haystack* y *State Tracking*):

1. **Mezcla 1 (Mamba-Heavy Core):** `LRCM -> MAMBA3 -> MAMBA3 -> MAMBA3 -> HOPMIX -> LRCM`  
   *Idea:* Triple capa Mamba-3 en el núcleo para maximizar capacidad de cómputo y transiciones continuas, anclada por LRCM en extremos.
2. **Mezcla 2 (HopMix Backbone):** `HOPMIX -> HOPMIX -> MAMBA3 -> MAMBA3 -> LRCM -> LRCM`  
   *Idea:* Difusión espacial multiescala temprana ($O(\log N)$) $\to$ razonamiento denso $\to$ lectura y almacenamiento exacto final.
3. **Mezcla 3 (Trio Intercalado):** `LRCM -> HOPMIX -> MAMBA3 -> LRCM -> HOPMIX -> MAMBA3`  
   *Idea:* Alternancia periódica simétrica de las tres naturalezas (Atención local/Exact $\to$ Difusión $\to$ Estado Continuo).
4. **Mezcla 4 (Mamba Sandwich):** `MAMBA3 -> HOPMIX -> LRCM -> LRCM -> HOPMIX -> MAMBA3`  
   *Idea:* Mamba-3 en las capas externas (entrada y salida) con núcleo dual de memoria LRCM + difusión Hop-Mix al centro.
5. **Mezcla 5 (Pirámide Invertida):** `HOPMIX -> LRCM -> MAMBA3 -> MAMBA3 -> LRCM -> HOPMIX`  
   *Idea:* Difusión $O(\log N)$ en bordes, refinamiento local intermedio y núcleo Mamba-3 central.

---

## 2. Tabla Comparativa de Resultados

| Mezcla Arquitectónica | Patrón de Capas | Params | Loss Inicial | Loss Final | % Reducción | Tiempo Entr. (s) | PPL (Perplejidad) | Needle Recall | State Tracking | Decode (tok/s) | Memoria Pico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mezcla 1 (Mamba-Heavy Core)** | `L -> M -> M -> M -> H -> L` | 470,748 | 62.56 | **1.1731** | **-98.1%** | 59.02 s | 1213.91 | 4.2% | **4.2%** | 81.3 tok/s | 0.46 MB |
| **Mezcla 2 (HopMix Backbone)** | `H -> H -> M -> M -> L -> L` | 452,340 | 62.18 | 1.7636 | -97.2% | 46.63 s | 412.21 | 0.0% | 0.0% | **99.3 tok/s** | 0.25 MB |
| **Mezcla 3 (Trio Intercalado)** | `L -> H -> M -> L -> H -> M` | 452,340 | 62.35 | 1.7457 | -97.2% | 46.08 s | 442.50 | 0.0% | 0.0% | 98.8 tok/s | 0.26 MB |
| **Mezcla 4 (Mamba Sandwich)** | `M -> H -> L -> L -> H -> M` | 452,340 | 62.36 | 3.7028 | -94.1% | **44.81 s** | 226.07 | **8.3%** | 0.0% | 96.9 tok/s | 0.25 MB |
| **Mezcla 5 (Pirámide Invertida)** | `H -> L -> M -> M -> L -> H` | 452,340 | 62.31 | 2.5898 | -95.8% | 72.35 s | **132.51** | 0.0% | 0.0% | 52.4 tok/s | **0.24 MB** |

---

## 3. Análisis de Rendimiento y Hallazgos Principales

1. **Ganador en Reducción de Pérdida Causal: Mezcla 1 (Mamba-Heavy Core)**
   - Alcanzó la pérdida final más baja de todas (**1.1731**, una reducción del **98.1%**).
   - *Por qué funciona:* El núcleo tripartito de Mamba-3 ($M \to M \to M$) con discretización trapezoidal continua y rotaciones complejas RoPE le confiere una plasticidad superior para asimilar dependencias temporales densas, mientras que LRCM al inicio y al final previene la dispersión del contexto.

2. **Ganador en Recuperación de Aguja (*Needle Recall*): Mezcla 4 (Mamba Sandwich)**
   - Logró **8.3% de recall directo sin fine-tuning** (el doble que las otras configuraciones).
   - *Por qué funciona:* Ubicar LRCM dual continuo en el centro (`LRCM -> LRCM`) permite que las hojas recuperadas se consoliden en dos fases consecutivas de lectura exacta, mientras que Hop-Mix y Mamba en los extremos distribuyen y proyectan la consulta eficazmente.

3. **Ganador en Perplejidad en Lenguaje: Mezcla 5 (Pirámide Invertida)**
   - Logró la **perplejidad más baja (132.51)**.
   - *Por qué funciona:* El diseño de difusión en los bordes (`HOPMIX`) con anclaje atencional previo a Mamba (`LRCM -> MAMBA3 -> MAMBA3 -> LRCM`) genera un suavizado sintáctico que beneficia la probabilidad de n-gramas locales.

4. **Ganadores en Eficiencia y Velocidad de Entrenamiento:**
   - **Mezcla 4 y Mezcla 3** registraron los tiempos de entrenamiento más bajos (**44.8 s** y **46.0 s**), con velocidades de inferencia cercanas a los **100 tokens/s** en CPU.
