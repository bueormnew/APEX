# Hop-Mix v1: especificación de un bloque de mezcla O(log N) guiado por atención

**Estado:** especificación completa + implementación de referencia verificada (`hopmix.py`, `test_hopmix.py`, 5 configuraciones, todos los tests pasan). **Pendiente para producción:** kernel fusionado (sección 10) y validación experimental (sección 11). Ambos están especificados con criterios de aceptación; ninguno está ejecutado todavía.

---

## 0. Definición y posición del bloque

> **Hop-Mix es un bloque de mezcla de información de coste sublineal por token, diseñado para operar junto a un mecanismo de atención y no en su lugar.** Cada token lee `S = K + 2r` posiciones de su pasado, con `K ≈ log₂ N` saltos geométricos fijos y `2r` posiciones sugeridas por la atención. Las combina con una compuerta normalizada y no exacta, y escribe el resultado en el flujo residual.
> 
> **La atención decide *a dónde* mirar. Hop-Mix reutiliza esas rutas muchas veces a bajo coste.**

Principios de diseño, en orden de prioridad:

1. **Velocidad pura.** Todo el bloque es `gather` + suma ponderada. Sin scans, sin chunks, sin recurrencias, sin proyecciones D×D por defecto.
2. **Una sola pasada, paralelismo total.** Cada token es independiente de los demás dentro de la capa. Causal por construcción.
3. **Complemento, no sustituto.** No hace recuperación exacta ni lo pretende. La precisión la aporta la atención; Hop-Mix aporta mezcla multiescala barata y propagación de las rutas que la atención ya calculó.
4. **Degradación elegante.** Sin rutas (`routes=None`) sigue siendo un bloque válido de saltos fijos. Con `init_scale=0` es exactamente la identidad, así que se puede insertar en un modelo preentrenado sin perturbarlo.

---

## 1. Definición formal

Sea `x ∈ ℝ^{N×D}` la entrada y `i` la posición de un token. Sea `n = RMSNorm(x)`.

**Memoria** (por token, calculada una sola vez):

```
m_j = n_j                 si mem_dim = None
m_j = W_down · n_j        si mem_dim = d_m          (W_down: D×d_m)
z_j = W_sal · n_j ∈ ℝ^H                              (salience, H = nº de grupos de compuerta)
```

**Conjunto de fuentes** `Src(i)` con `S = K + 2r` ranuras:

| Tipo | Ranuras | Índice | Origen |
| --- | --- | --- | --- |
| Fija | K | `i − h_k`, con `h_k = 1, 2, 4, …, 2^{K−1}` | geometría |
| Guiada | r | `R[i, m]` | top-r de la atención (capa anterior) |
| Pointer-jump | r | `R[R[i, m], 0]` | composición de rutas |

**Validez** de la ranura `s` con índice `j`:

```
valid(i, s) = (0 ≤ j ≤ i) ∧ tok[j] ∧ (sid[j] = sid[i])
```

Las tres condiciones garantizan, respectivamente, causalidad estricta, exclusión de padding y aislamiento entre secuencias empaquetadas. Un índice inválido (`−1`, futuro, o `j` en otra secuencia) se enmascara y nunca contribuye.

**Compuerta** (por grupo de canales `h = 1..H`, en fp32):

```
ℓ[i,h,s] = (W_g · n_i)[h,s]              término de consulta (token destino)
         + b[h,s]                         sesgo por fuente
         + z_{src(i,s)}[h]                salience del token fuente (contenido de la fuente)
         + D_s[h, bucket(i − j)]          sesgo por distancia log₂ (solo ranuras guiadas)
         + β_s · log w[i,s]               peso de atención de la ruta (ranuras guiadas)

g[i,h,·] = softmax( [ ℓ[i,h,s] enmascarado ,  ν_h ] )[0:S]     ν_h = logit "nulo"
```

El logit nulo `ν_h` es un sumidero aprendido: permite que el bloque «no escriba nada», y garantiza que la suma de la compuerta sea ≤ 1 y que ninguna fila quede sin masa (sin NaN aunque no haya fuentes válidas, p. ej. el token 0).

**Mezcla** (canales divididos en H grupos de `d_h = d_m / H`):

```
y_i[h] = Σ_s  g[i,h,s] · ( a_s[h] ⊙ m_{src(i,s)}[h] )         a_s ∈ ℝ^{S×d_m}
```

**Salida:**

```
out_i = x_i + γ ⊙ ( W_up · y_i )       si mem_dim ≠ None
out_i = x_i + γ ⊙ y_i                  si mem_dim = None
γ ∈ ℝ^D, inicializado a init_scale (1e-2 por defecto; 0 = identidad exacta)
```

---

## 2. Qué aporta cada pieza (y qué se descartó)

| Pieza | Por qué está | Coste |
| --- | --- | --- |
| Saltos `2^k` | cobertura multiescala sin depender de nadie | 0 parámetros, lectura contigua |
| Rutas de atención | contenido: lleva información a donde importa | gather aleatorio |
| Pointer-jump `R[R[i]]` | duplica la profundidad de alcance con una lectura | 1 gather de enteros extra |
| Salience `z_j` | la compuerta ve el *contenido de la fuente*, no solo el destino | `N·D·H` MAC, 1 vez |
| Sesgo por distancia | distingue una ruta cercana de una lejana | tabla pequeña |
| `β·log w` | **da gradiente a la atención** a través de las rutas (ver §6.3) | escalar |
| Logit nulo | estabilidad numérica y capacidad de abstenerse | H parámetros |
| `mem_dim` | reduce caché y tráfico de memoria; mezcla en espacio comprimido | `2·D·d_m` MAC |
| Dropout de fuentes | evita que el bloque dependa de un solo salto | 0 en inferencia |

**Descartado deliberadamente:** compuerta dependiente de la clave por producto punto completo (sería atención con otro nombre), scans/cumsum/chunks (violan la restricción de una sola pasada), scatter en el forward, proyección D×D por defecto (es lo único que domina el coste, ver §5.2).

---

## 3. Cobertura y propagación de información

**Proposición.** Con saltos `{1, 2, 4, …, 2^{K−1}}`, el token `i` puede recibir información del token `j = i − d` (0 \< d \< 2^K) tras `popcount(d) ≤ K` capas de Hop-Mix, sin usar rutas guiadas.

*Esbozo:* `d` es suma de las potencias de 2 de su representación binaria; cada capa permite un salto de la forma `2^k`, así que se compone un camino de `popcount(d)` capas. (Verificado exhaustivamente para `d < 128` en `test_coverage`.)

**Consecuencias honestas:**

- La existencia de un camino **no** implica recuperación: cada salto atraviesa una compuerta softmax y una suma, así que la señal se atenúa y se mezcla. Es el comportamiento buscado (mezcla borrosa), no una limitación a «arreglar».
- Con rutas guiadas y pointer-jump, un token alcanza un destino arbitrario en 1–2 capas **si la atención lo señaló**. Esa es la razón de ser del diseño híbrido.
- Para diversificar la cobertura entre capas, usa conjuntos de saltos distintos por capa (`make_hops(max_len, base, offset)`, p. ej. alternar base 2 y base 3).

---

## 4. Parámetros por capa

Con `S = K + 2r`, `H` grupos, `dm = d_m ó D`:

| Tensor | Tamaño |
| --- | --- |
| `W_g` | `D · H · S` |
| `W_sal` | `D · H` |
| `a` | `S · dm` |
| `b`, `ν`, `D_s`, `β` | `H·S + H + 2r·buckets·H + 2r` (despreciable) |
| `γ`, `RMSNorm` | `2D` |
| `W_down`, `W_up` | `2 · D · d_m` (solo con `mem_dim`) |

Ejemplo: `D=1024, N=128K → K=17, r=2, S=21, H=4`, sin `mem_dim`: ≈ `86K + 4K + 21K + 2K ≈ 113K` parámetros por capa (una capa de atención con proyecciones tiene `4D² ≈ 4.2M`).

---

## 5. Complejidad: la cuenta honesta

### 5.1 Lecturas y cómputo del mezclador, por token

|  | Atención | Hop-Mix (`mem_dim=None`) |
| --- | --- | --- |
| Lecturas de vectores | N | S = K + 2r |
| MAC de mezcla | ≈ 2·N·D (media causal: N·D) | S·D |
| MAC de compuerta | — | D·H·S + D·H |
| Proyecciones D×D | 4D² | 0 (opcional: `2·D·d_m`) |

Cifras con `D=1024, H=4, r=2`:

| N | S | Mezcla de atención (media causal, N·D) | Hop-Mix total (compuerta + mezcla) | Cociente |
| --- | --- | --- | --- | --- |
| 128 | 11 | 131 K | ≈ 60 K | **≈ 2×** |
| 4 096 | 16 | 4.2 M | ≈ 86 K | ≈ 49× |
| 32 768 | 19 | 33.5 M | ≈ 101 K | ≈ 330× |

### 5.2 Lo que esto significa para tu objetivo

- **A N=128 el cociente del mezclador es modesto (\~2×).** A esa escala la atención está dominada por sus proyecciones `4D²` (≈ 4.2 M MAC), no por la mezcla. Un Hop-Mix **sin proyecciones D×D** frente a un bloque de atención completo sí es ≈ 70× menos MAC, pero ese ahorro viene de *no tener proyecciones*, no de log N.
- **El factor log N brilla con N grande.** El coste de Hop-Mix casi no crece con N; el de la atención crece linealmente por token.
- **El tiempo real no sigue a los MAC.** Hop-Mix está limitado por ancho de banda de memoria (gather), no por cómputo. Las lecturas de saltos fijos son contiguas y cacheables en L2; las de rutas guiadas son aleatorias. **Ningún número de latencia aquí es medido**; el criterio de aceptación está en §11.3.
- Si añades `W_down/W_up` el coste sube a `2·D·d_m` MAC por token. Con `d_m = D/4` son `D²/2`, 8× menos que las 4D² de la atención, pero deja de ser «casi gratis». Elígelo si la caché o el tráfico de memoria te preocupan más que los MAC.

### 5.3 Memoria de caché de inferencia

Hop-Mix guarda, por token y por capa: `m_j` (`dm` valores), `z_j` (H fp32), `routes` (r enteros) y `rw` (r fp32).

- **Comparación honesta:** con atención GQA, el KV-cache puede ser *menor* que `D` por token. Hop-Mix sin `mem_dim` guarda `D`. Con `mem_dim = D/4` guarda `D/4`, menos que la mayoría de configuraciones de KV.
- **No se puede desalojar tokens antiguos:** el salto `2^k` accede a *cualquier* posición pasada algún día. Si la memoria es crítica, usa `mem_dim` o restringe `max(hops)` a una ventana (a costa de cobertura lejana, que cubren las rutas).

---

## 6. Interfaz con la atención

### 6.1 Qué entrega la atención

Por token: `routes ∈ ℤ^{r}` (índices causales, `−1` si no hay) y `route_w ∈ ℝ^{r}` (masa de atención de cada ruta).

### 6.2 Cómo se obtienen

- **Cabezas guía.** Designa `G` cabezas (típicamente 1–2) que emiten rutas; promedia sus probabilidades y toma top-r (`routes_from_attention`). No uses todas las cabezas: encarece sin mejorar.
- **Con FlashAttention la matriz N×N no existe.** El kernel debe mantener un top-r en línea por fila (fusión de top-r por bloque, sobre los scores `q·k` ya calculados, normalizando con el `logsumexp` final). Coste adicional esperado: pequeño frente al propio kernel; **debe medirse** (§11.3).
- **Alternativas** si no se puede tocar el kernel de atención: tomar rutas de una capa de atención de ventana corta, o de un enrutador diminuto (`d_r ≈ 16`) con su propio top-r.

### 6.3 Cómo aprende la atención a guiar bien

Los índices no son diferenciables, así que por sí solos la atención no recibiría señal de si sus rutas ayudaron. La solución es el término `β_s · log w`: `route_w` conserva el gradiente hacia las probabilidades de atención. Si una ruta ayuda, el gradiente sube su probabilidad; si estorba, la baja. Verificado en `test_grads` (el gradiente hacia `route_w` es no nulo y finito).

### 6.4 Reutilización

Las mismas `routes` alimentan a las `n_mix` capas Hop-Mix siguientes; la siguiente capa de atención las renueva. Una capa de atención más un grupo de `3–5` Hop-Mix es el punto de partida.

---

## 7. Arquitectura híbrida recomendada

```
[Atención (G cabezas guía)] → routes, route_w
    [Hop-Mix] × n_mix   (reusan routes)
[MLP]  (entre capas, como siempre)
[Atención] → renueva routes
    ...
```

| Decisión | Valor inicial | Cómo ajustarlo |
| --- | --- | --- |
| Proporción atención : Hop-Mix | 1 : 3 a 1 : 5 | subir Hop-Mix hasta que caiga la calidad en recuperación |
| Atención restante | completa en las capas profundas, o ventana + sumideros | probar ambas |
| `r` | 2 | 1–4; más rutas = más gathers aleatorios |
| `H` | 4 | 2–8 |
| `mem_dim` | `None` | `D/4` si importa la caché |
| `hops` por capa | alternar base 2 / base 3 | ablar |
| Posición | tras la atención que emite las rutas | probar al inicio de cada grupo |

El MLP entre capas aporta la mezcla de canales que Hop-Mix no hace (solo escala por canal). Por eso `W_out` D×D no es necesaria por defecto.

---

## 8. Entrenamiento

1. **Inicialización:** `γ = init_scale (1e-2)` para entrenar desde cero, `0` para insertar en un modelo preentrenado (identidad exacta, verificada). `W_g`, `W_sal` \~ N(0, 0.02²); `a = 1`; sesgos a 0.
2. **Normalización:** RMSNorm de entrada; mezcla en el espacio normalizado; la escala de salida la fija `γ`.
3. **Precisión:** compuerta y acumulación en fp32; valores y caché en bf16/fp16. El enmascarado usa `−1e30` (finito) para evitar NaN.
4. **Dropout de fuentes:** `p ≈ 0.05–0.1` durante el entrenamiento, para evitar la dependencia de un único salto.
5. **Backward:**
   - Saltos fijos: el backward es una **suma desplazada** (determinista, sin atómicos).
   - Rutas guiadas y pointer-jump: el backward de un gather es un *scatter-add*. Opciones: `atomicAdd` fp32 (rápido, no determinista) o ordenar por destino (determinista, más lento). Ofrece ambos modos (`deterministic=True/False`).
   - El forward no usa scatter.
6. **Memoria de activaciones:** nunca materializar `[B,N,S,D]`. El kernel debe recomputar compuerta y gathers en el backward.
7. **Receta de modelo híbrido:**
   - Desde cero: entrenar el híbrido completo con las proporciones de §7.
   - Adaptación: insertar Hop-Mix con `γ=0` en un modelo preentrenado, congelar el resto unas pocas miles de pasos, luego descongelar y *reducir gradualmente* capas de atención (destilación opcional desde el modelo original).

---

## 9. Inferencia incremental

`HopMix.step(x_t, t, cache, routes_t, route_w_t, pad_t, sid_t)` procesa un token usando solo el estado de la caché.

- **Garantía:** la salida coincide con el forward paralelo hasta error numérico (`< 1e-5` en fp32). Verificada para las 5 configuraciones de la suite, con padding y secuencias empaquetadas activos.
- **Coste por token generado:** `S` lecturas + la compuerta; **independiente de N**.
- **Requisito:** la caché guarda también las rutas de cada token (necesarias para el pointer-jump).

---

## 10. Especificación del kernel fusionado (Triton/CUDA)

El código de referencia define la semántica; el kernel debe igualarla.

**Forward** — un programa por `(b, bloque de tokens, bloque de canales)`:

1. Calcular los índices de las S fuentes en registros (saltos fijos por aritmética; guiadas por carga de `routes`; pointer-jump por una carga de enteros adicional).
2. Aplicar validez (causal, padding, `sid`).
3. Cargar `q = W_g·n_i`, los `z_j` por gather y los sesgos; softmax en fp32 incluido el logit nulo.
4. Acumular `Σ g · a_s ⊙ m_src` en fp32 recorriendo las S fuentes. Los saltos fijos se leen como flujos contiguos.
5. Fusionar residual y `γ`.

**Backward** — recomputa la compuerta; para saltos fijos usa suma desplazada; para guiadas usa scatter-add (modo atómico o determinista).

**Aceptación del kernel:**

- Igualdad numérica con la referencia (tolerancia fp32 `1e-5`, bf16 `1e-2` relativa) para forward y todos los gradientes.
- Sin materialización de `[B,N,S,D]` (verificable con el perfilador de memoria).
- Los 5 tests de `test_hopmix.py` pasan con el kernel sustituido.

---

## 11. Plan de validación

### 11.1 Tests funcionales (ya implementados y pasando en la referencia)

Causalidad estricta, paridad paralelo/incremental, aislamiento de secuencias empaquetadas y de padding, gradientes finitos (incluido hacia la atención), casos límite (N=1, sin rutas, N > max_len), identidad con `init_scale=0`, pila híbrida causal, cobertura de saltos.

### 11.2 Experimentos de calidad (no ejecutados)

| Experimento | Pregunta | Criterio de éxito sugerido |
| --- | --- | --- |
| Copia / recuerdo asociativo multi-consulta | ¿Hop-Mix + poca atención iguala a atención completa? | ≥ 95 % de la exactitud de atención completa con ≤ 1/4 de capas de atención |
| Needle-in-a-haystack | ¿Las rutas guiadas rescatan la recuperación lejana? | recuperación > baseline sin rutas |
| Perplejidad en modelo pequeño | ¿Coste de calidad? | ≤ +2 % de perplejidad con ≥ 2× de ahorro de tiempo de mezcla |
| Extrapolación de longitud | ¿Los saltos geométricos generalizan? | degradación suave a 2× la longitud de entrenamiento |

**Ablaciones:** sin rutas / sin pointer-jump / sin salience / sin sesgo de distancia / sin `β·log w` / sin logit nulo / `mem_dim` ∈ {None, D/2, D/4} / base 2 vs 3 / `r` ∈ {0,1,2,4} / proporción atención:Hop-Mix.

### 11.3 Benchmark de velocidad (no ejecutado)

- Medir **tiempo de pared** (no MAC) a N ∈ {128, 1k, 8k, 32k, 128k}, forward y forward+backward, bf16, en el hardware objetivo.
- Comparar con FlashAttention y con una capa de atención completa (con proyecciones).
- Medir por separado el coste de emitir rutas desde el kernel de atención.
- **Criterio sugerido:** el mezclador debe ser ≥ 3× más rápido que FlashAttention a N=4k y la ventaja debe crecer con N. Si no se alcanza, lo primero a revisar es el patrón de acceso de los gathers guiados.

---

## 12. Riesgos y límites conocidos

1. **No es recuperación exacta.** Por diseño. Cualquier tarea que exija localizar un token concreto depende de la atención.
2. **Gain a N pequeño es limitado** (§5.2). El diseño apunta a contextos largos.
3. **Memoria limitada por ancho de banda.** El ahorro teórico en MAC no se traduce 1:1 en tiempo.
4. **Backward no determinista** en modo atómico.
5. **Dependencia de la calidad de las rutas.** Si la atención guía mal, el beneficio se reduce a los saltos fijos. Mitigación: el término `β·log w` y las ablaciones de §11.2.
6. **Caché sin desalojo** (§5.3).
7. **Emitir top-r desde FlashAttention requiere modificar el kernel**, que es el punto de integración más delicado.
8. **Los resultados de calidad y velocidad no están medidos.** Esta especificación fija el diseño y los criterios; no afirma que se cumplan.

---

## 13. Configuración por defecto

| Parámetro | Valor |
| --- | --- |
| `hops` | `1, 2, 4, …, < max_len` |
| `n_routes (r)` | 2 |
| `pointer_jump` | `True` |
| `n_gate_heads (H)` | 4 |
| `mem_dim` | `None` |
| `use_salience` | `True` |
| `n_dist_buckets` | 24 |
| `init_scale` | `1e-2` (`0` para inserción en modelo preentrenado) |
| `source_dropout` | `0.0` en el código; recomendado `0.05` al entrenar desde cero |
| `norm_eps` | `1e-6` |

---

## 14. Archivos

- `hopmix.py`: implementación de referencia (`HopMix`, `HopMixConfig`, `HopMixCache`, `routes_from_attention`, `AttnWithRoutes`, `HybridStack`).
- `test_hopmix.py`: suite de tests (ejecutar con `python test_hopmix.py`).