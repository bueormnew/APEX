# Arquitecturas Implementadas: HOP-MIX, LRCM, MAMBA-3 y ECHO

Este documento resume la implementación técnica, matemática y funcional de la suite de arquitecturas solicitada, desarrollada en PyTorch con compatibilidad causal autoregresiva completa de punta a punta (*end-to-end*).

---

## 1. ECHO (Engrama Compresivo de Alta Ortogonalidad) — [`echo.py`](file:///c:/Users/gerso/Desktop/IA-REAL/echo.py)
ECHO sustituye a los FFNs densos tradicionales, aportando direccionabilidad por contenido y una enorme capacidad de memorización sin disparar los parámetros.
- **Product Key Memory:** La consulta normalizada $q \in \mathbb{R}^{d}$ se divide en dos mitades $q_1, q_2 \in \mathbb{R}^{d/2}$. Cada mitad se compara contra tablas de claves $K_1, K_2 \in \mathbb{R}^{n_{keys} \times (d/2)}$, generando un espacio composicional de $n_{keys}^2$ combinaciones pagando únicamente $2 \cdot n_{keys}$ comparaciones de producto punto.
- **Micro-Expertos Factorizados de Bajo Rango ($r$):** Cada clave selecciona matrices factorizadas $W_{down} \in \mathbb{R}^{d \times r}$ y $W_{up} \in \mathbb{R}^{r \times d}$. Cada combinación ejecuta:
  $$\text{Expert}_i(h) = W_{up, i} \cdot \text{SiLU}(W_{down, i} \cdot h)$$
  y se ponderan suavemente con la distribución conjunta de softmax marginal.
- **Estabilidad y Mitigación de Colapso de Expertos:**
  - Enrutamiento suave con temperatura $\tau$.
  - Pérdida auxiliar de balanceo de carga: penaliza la dispersión/varianza del tráfico entre claves por batch con coeficiente $\alpha \approx 0.01$.
  - Métrica de entropía normalizada monitoreada para prevenir claves muertas.
- **Compresor Compartido:** Cuello de botella $W_{o2}(\text{SiLU}(W_{o1}(\text{RMSNorm}(m))))$, escalado con las combinaciones activas $k \times k$.
- **Compuerta SiLU:** $\text{out} = W_{out} \cdot (\text{SiLU}(W_{gate} \cdot h) \odot m_{final})$.

---

## 2. HOP-MIX v1 con ECHO Nativo — [`hopmix.py`](file:///c:/Users/gerso/Desktop/IA-REAL/hopmix.py)
Mecanismo de mezcla de información causal de coste sublineal $O(\log N)$ por token, guiado por atención y complementado nativamente con ECHO en lugar de FFN:
- **Saltos Geométricos Fijos ($K \approx \log_2 N$):** Fuentes $i - 2^k$ para cobertura multiescala sin costo paramétrico.
- **Ranuras Guiadas de Atención ($r$) + Pointer-Jumps:** Conexiones directas a las posiciones señaladas por atención previa $R[i, m]$ y doble alcance composicional $R[R[i, m], 0]$.
- **Compuerta Causal Normalizada:**
  $$\ell[i, h, s] = (W_g \cdot n_i)[h, s] + b[h, s] + z_{src(i,s)}[h] + D_s[h, \text{bucket}(i-j)] + \beta_s \cdot \log w[i, s]$$
  con sumidero de logit nulo $\nu_h$ que permite que el bloque se abstenga y previene NaNs.
- **Sustitución de FFN:** El bloque `HopMixECHOBlock` procesa la mezcla contextual y aplica de forma directa y nativa el bloque `ECHO`.
- **Inferencia Autoregresiva Causal:** Soporta el método `.step(x_t, cache)` en tiempo constante $O(S)$ por token generado con `HopMixCache`.

---

## 3. LRCM con ECHO Nativo — [`lrcm.py`](file:///c:/Users/gerso/Desktop/IA-REAL/lrcm.py)
Arquitectura de reemplazo de atención que desacopla la mezcla local de la recuperación histórica de largo alcance:
- **Exact Local Mixing:** Atención causal exacta sobre ventana reciente $W$ ($W=16..256$).
- **Jerarquía Convolucional Multiescala:** 
  $$\text{Tokens} \to \text{Chunk Descriptors} \to \text{Page Descriptors} \to \text{Region Descriptors}$$
  utilizando convoluciones causales depthwise 1D para propagación temporal sin fugas al futuro.
- **Hierarchical Learned Routing:** La consulta $q_{mem} = W_q [x_t; h_{local}]$ evalúa los descriptores de manera causal mediante *sparse beam search* (top-$B$).
- **Exact Leaf Gather + Tiny Exact Attention:** Una vez seleccionadas las mejores hojas históricas (chunks de tokens reales), se extrae su representación guardada en memoria y se calcula atención exacta sobre ese fragmento:
  $$O_{recall} = \text{softmax}\left(\frac{q_{recall} K_{exact}^T}{\sqrt{d}}\right) V_{exact}$$
- **Fusión Normalizada:** Compuerta aprendida de softmax entre la rama local y la rama de recall exacto.
- **Sustitución de FFN:** Integración del bloque `LRCMECHOBlock`, donde la salida fusionada fluye de inmediato a `ECHO`.

---

## 4. MAMBA-3 Puro — [`mamba3.py`](file:///c:/Users/gerso/Desktop/IA-REAL/mamba3.py)
Implementación matemáticamente rigurosa según la formulación de *Dao & Gu (arXiv:2603.15569)*, integrando todas las ventajas sobre Mamba-2:
1. **Discretización Exponencial-Trapezoidal de 2º Orden:**
   Sustituye la aproximación de orden cero (Euler/ZOH) de Mamba-2 promediando el aporte de entrada $(B \cdot x)_{t-1}$ y $(B \cdot x)_t$ mediante una compuerta aprendida $\text{trap}_t = \sigma(W_{trap} x + b_{trap})$:
   $$u_t = (1 - 0.5 \cdot \text{trap}_t) (B_t \cdot x_t) + (0.5 \cdot \text{trap}_t) (B_{t-1} \cdot x_{t-1})$$
   $$h_t = \exp(\Delta_t A) h_{t-1} + \Delta_t u_t$$
2. **Espacio de Estados Complejos con RoPE Dinámico:**
   Se aprende un ángulo por cabeza $\Delta \theta_t = \Delta_t \cdot \text{angle\_proj}(x_t)$. Se rotan $B_t$ y $C_t$ con matrices de rotación 2D sobre pares contiguos (isomorfo a $e^{i \theta}$ en $\mathbb{C}$), otorgando capacidad para resolver paridad y seguimiento de estados sin números complejos explícitos.
3. **Formulación MIMO (Multi-Input Multi-Output):**
   La entrada se expande con rango $R$ (`mimo_rank`), permitiendo un flujo de mayor intensidad aritmética que convierte las operaciones de memoria en operaciones intensivas en cómputo durante la decodificación.
4. **Modo Recurrente Incremental:** Soporta generación eficiente token a token con persistencia de memoria trapezoidal `bx_prev` en `Mamba3State`.

---

## 5. Modelo Causal Híbrido Unificado — [`hybrid_model.py`](file:///c:/Users/gerso/Desktop/IA-REAL/hybrid_model.py)
El modelo `HybridCausalLM` permite componer y alternar libremente los tres bloques con una interfaz end-to-end homogénea:
- **Patrón Modular Configurable:** Por ejemplo `LRCM -> HOP -> MAMBA -> MAMBA -> HOP -> LRCM -> ...`
- **Entrenamiento:** Forward paralelizado con cálculo de CrossEntropy causal sobre logits desplazados $+ \text{aux\_loss}$ de ECHO.
- **Inferencia Autoregresiva:** Método `.generate(prompt, max_new_tokens, temperature, top_k)` que sincroniza de forma transparente las cachés de HopMix, estados de memoria jerárquica de LRCM y estados recurrentes de Mamba-3.
