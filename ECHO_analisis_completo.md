# ECHO (Engrama Compresivo de Alta Ortogonalidad): planteamiento, pruebas y recomendaciones

**Resumen ejecutivo:** ECHO propone reemplazar el FFN de un transformer por un bloque de memoria direccionable por contenido (product keys + micro-expertos factorizados + compresor compartido), entrenable end-to-end con next-token loss, para que un modelo pequeño "almacene" muchos más hechos que su tamaño denso equivalente. Implementé el bloque completo (la parte que en el diseño original quedaba sin terminar), lo sometí a 6 baterías de pruebas en CPU a escala diminuta, y encontré: **la idea central funciona y es real**, con una ventaja de capacidad notable en recall de datos sesgados (el escenario más parecido al lenguaje real), pero **las afirmaciones de rendimiento extremo no están sustentadas**, tiene debilidades concretas (regresión suave sin estructura de "hechos", velocidad real, y un intento de híbrido que empeoró en vez de mejorar), y el riesgo de colapso de expertos es real bajo ciertas condiciones de enrutamiento no probadas aquí.

---

## 1. Planteamiento original

La propuesta parte de una v1 (no evaluada aquí) descrita como "potente pero sucia": fases de entrenamiento separadas, un componente KAN lento, y un loop recurrente no paralelizable. ECHO (v2) se plantea como corrección con tres reglas de diseño:

1. Entrenamiento end-to-end desde cero, con únicamente la pérdida estándar de siguiente-token.
2. Una sola pasada hacia adelante, 100% paralelizable (sin recurrencia).
3. Implementación mínima (menos de 80 líneas de código).

La idea central, explícitamente inspirada en **Product Key Memory** (Lample et al., Facebook, 2019) y **PEER** (DeepMind, 2024): en vez de guardar hechos como vectores estáticos, cada "slot" de memoria es una función diminuta (un micro-experto de 2 capas). Como cada micro-experto se reutiliza en muchas combinaciones distintas, el argumento es que el bloque no puede simplemente memorizar sino que se ve forzado a aprender funciones reutilizables — de ahí la promesa de mejor generalización, no solo más capacidad.

## 2. Cómo funciona ECHO — arquitectura completa

Entrada: `h` de forma `[B, T, d]` (salida de la capa de atención en un transformer).

### 2.1 Consulta rápida — Product Keys
```
q = RMSNorm(W_q @ h)          # [B, T, d]
q1, q2 = q.chunk(2)           # se parte en dos mitades, cada una de d/2
score1 = q1 @ K1.T            # K1: [n_keys, d/2] -> n_keys puntuaciones
score2 = q2 @ K2.T            # K2: [n_keys, d/2] -> n_keys puntuaciones
```
Se toman los top-k de cada lado. El producto cartesiano de los índices elegidos da acceso a `n_keys²` combinaciones posibles pagando solo el costo de `2·n_keys` comparaciones — la parte del diseño que realmente compra capacidad casi gratis.

### 2.2 Recuperación composicional — el truco de generalización
Cada combinación no recupera un vector fijo; ejecuta dos micro-expertos de bajo rango (`r`) y los suma:
```
Expert_i(h) = W_up_i @ SiLU(W_down_i @ h)          # rango r, no rango d
m = Σ_i p_i · (Expert_{k1_i}(h) + Expert_{k2_i}(h))
```
`W_down_i`/`W_up_i` son tablas factorizadas (una por lado), no una tabla por cada una de las `n_keys²` combinaciones — eso es lo que mantiene el conteo de parámetros bajo control.

### 2.3 Compresor compartido — el cuello de botella
Una sola capa, compartida por todo el bloque, reemplaza el loop recurrente de la v1:
```
m_final = m + W_o2 @ SiLU(W_o1 @ RMSNorm(m))       # bottleneck << d
```

### 2.4 Salida e integración en la capa
```
out = W_out @ (SiLU(W_gate @ h) * m_final)

Transformer Layer (versión ECHO):
  h = h + Attention(RMSNorm(h))
  h = h + ECHO(RMSNorm(h))        # sustituye al FFN, no va en paralelo
```

### 2.5 Afirmaciones de la propuesta original (sin verificar en el texto fuente)
- Complejidad: `O(B·T·d²)` de un FFN normal (~4M ops/token con d=2048) vs `O(B·T·(√N·d + k·r·d))` de ECHO (~0.8M ops/token) → "5x más barato".
- Capacidad efectiva estimada: "~14-18B de hechos equivalentes" para un modelo de 2.1B parámetros reales.
- "2B con ECHO > 7B denso en razonamiento (MMLU, GSM8K)".

Ninguna de estas tres cifras venía acompañada de una derivación, ni de un experimento, ni de una cita — son extrapolaciones de vuelta de hoja. El resto de este documento reporta lo que **sí** medí.

---

## 3. Metodología del benchmark

Todas las pruebas corrieron en un contenedor Linux sin GPU, usando **JAX (CPU)** en vez de PyTorch (PyTorch desde PyPI arrastra dependencias CUDA obligatorias de varios GB que no cupieron en el disco disponible del entorno). Terminé la parte del código que el mensaje original dejaba sin implementar (el `gather` de expertos + la suma ponderada, marcada como `# ...` en el esqueleto original).

**Regla de oro seguida en todas las pruebas: paridad exacta de parámetros.** Cada comparación FFN-vs-ECHO usa el mismo número total de parámetros en el bloque (verificado programáticamente, no aproximado), para que ninguna arquitectura gane simplemente por tener más presupuesto.

Se probaron 4 escenarios de tarea, cada uno diseñado para aislar una hipótesis distinta:

| # | Tarea | Qué aísla |
|---|---|---|
| 1 | Recall asociativo (Hopfield-style, sin cabezal libre) | Capacidad pura de memorización |
| 2 | Recall con muestreo sesgado Zipf | Interferencia catastrófica / colapso de expertos, escenario realista tipo lenguaje |
| 3 | Regresión contra función maestra aleatoria | Aproximación de función suave y global, SIN estructura de "hechos discretos" |
| 4 | Velocidad real (forward+backward) | Costo de cómputo real, no FLOPs teóricos |
| 5 | Interpolación en una variedad continua | Generalización a puntos nunca vistos |
| 6 | Híbrido (bloques apilados con residual) | Si intercalar ECHO+FFN da "lo mejor de ambos mundos" |

⚠️ **Nota metodológica importante:** mi primer intento de la Tarea 1 tenía un defecto — usaba un clasificador lineal final entrenable (`d × n_hechos` parámetros) que por sí solo podía memorizar la tarea sin que el bloque hiciera nada útil, ocultando cualquier diferencia real entre arquitecturas. Lo corregí exigiendo que el bloque reproduzca directamente un vector objetivo fijo (sin cabezal entrenable de por medio) — este es el diseño que reporto abajo. Lo menciono porque es exactamente el tipo de error que infla resultados en benchmarks de "memoria" sin que nadie lo note.

---

## 4. Resultados — tablas con números reales

### 4.1 Recall asociativo puro (datos uniformes, sin cabezal libre)
Presupuesto: 78,848 parámetros en ambos. Full-batch, 400 pasos.

| N_HECHOS | FFN denso | ECHO |
|---|---|---|
| 300 | 100.0% | 100.0% |
| 1,000 | 100.0% | 100.0% |
| 2,500 | 99.9% | 100.0% |
| **5,000** | **76.9%** | **98.4%** |
| **10,000** | **22.4%** | **64.3%** |

Uso de claves (colapso): entropía normalizada 0.99-1.00 en todos los casos — **sin colapso** en este régimen de enrutamiento suave (softmax continuo).

### 4.2 Recall con muestreo sesgado tipo Zipf (s=1.3) — el escenario realista
3,000 hechos, 1,500 pasos, batch=256, mismo presupuesto (78,848).

| Escenario | FFN | ECHO |
|---|---|---|
| Uniforme — recall total | 13.5% | 100.0% |
| Zipf — recall total | 5.5% | 47.5% |
| Zipf — 10% más frecuentes | 52.7% | 100.0% |
| **Zipf — 50% más raros (cola)** | **0.2%** | **15.2%** |

Colapso bajo sesgo: entropía baja de ~0.97 (uniforme) a ~0.83-0.87 (Zipf); la clave más usada pasa de acaparar ~3% del tráfico a ~10-12%. **Desbalance real y medible, pero no colapso total** (las 64 claves de cada lado siguen usándose todas).

### 4.3 Regresión contra función suave y global (sin estructura de "hechos")
Función objetivo generada por una red maestra aleatoria fija (sin clusters ni hechos discretos). Presupuesto ajustado a distintos tamaños para forzar el régimen de escasez de parámetros.

| Params (ambos) | Complejidad de la función maestra | FFN (test) | ECHO (test) | Ganador |
|---|---|---|---|---|
| 78,848 | hidden=128 (fácil) | 99.99% | 99.96% | empate |
| 17,920 | hidden=512 | 99.97% | 99.77% | FFN (leve) |
| **11,008** | **hidden=1024 (difícil)** | **99.95%** | **99.00%** | **FFN (20x menos error)** |

### 4.4 Velocidad real (forward + backward, CPU, mismos parámetros)
78,848 parámetros en ambos.

| Batch | FFN | ECHO | Overhead de ECHO |
|---|---|---|---|
| 32 | 0.18 ms | 0.25 ms | +37% |
| 256 | 1.32 ms | 1.73 ms | +30% |
| 1,024 | 4.78 ms | 7.44 ms | +56% |

El `gather` disperso sobre las tablas de expertos es una operación memory-bound; el ahorro teórico de FLOPs no se traduce en velocidad real sin un kernel dedicado (ver §6.2).

### 4.5 Interpolación (generalización a puntos nunca vistos)
Variedad continua (círculo), 24 puntos de entrenamiento, evaluado en los puntos medios.

| Configuración | FFN (interp.) | ECHO (interp.) |
|---|---|---|
| n_keys=64, top-4 (blending suave) | 93.4% | 94.8% |
| n_keys=4, top-1 (ruteo duro, sin blending) | 93.4% | 94.7% |

No encontré aquí una debilidad de ECHO — incluso con ruteo duro top-1, interpola tan bien como el FFN en esta tarea (la función objetivo resultó ser una relación simple y global entre las mismas coordenadas de entrada, aprendible por igual por cualquiera de los dos expertos elegidos).

### 4.6 Híbrido (ECHO→FFN apilado con residual) — misma prueba, mismo presupuesto total

| Prueba | FFN puro | ECHO puro | **Híbrido** |
|---|---|---|---|
| Recall Zipf — total | 5.2% | **47.5%** | 26.8% |
| Recall Zipf — cola rara | 0.1% | **15.2%** | 2.1% |
| Regresión — test | **99.97%** | 99.93% | 99.85% |
| Velocidad (ms/paso) | **4.6 ms** | 6.7 ms | 8.2 ms |

El híbrido **no combina las fortalezas — las diluye.** Al partir el presupuesto entre dos sub-bloques, cada uno queda más pequeño que su versión pura, y en velocidad simplemente suma ambos costos. El orden de apilado (ECHO→FFN vs FFN→ECHO) también importa y ninguno superó al especialista puro correcto para cada tarea.

---

## 5. Fortalezas confirmadas

1. **Ventaja de capacidad real bajo datos sesgados tipo Zipf** (el escenario más parecido al lenguaje real): hasta 75x más recall en la cola de hechos raros frente a un FFN con los mismos parámetros. Es la evidencia más sólida a favor del diseño.
2. **Mecánica de implementación válida:** el producto cartesiano de product-keys + expertos factorizados de bajo rango es diferenciable de punta a punta, entrena con cross-entropy/coseno estándar, sin fases separadas — la promesa de "end-to-end, una sola pasada" se sostiene en la práctica.
3. **Resistencia parcial a interferencia catastrófica:** con datos sesgados, el FFN denso pierde casi toda la cola rara (0.2%) mientras ECHO retiene una fracción sustancial (15.2%) — consistente con la intuición de que el enrutamiento disperso evita que ejemplos frecuentes "pisen" el gradiente de los raros.

## 6. Debilidades confirmadas a pequeña escala

1. **Pierde en regresión suave sin estructura de "hechos":** cuando la tarea no tiene clusters discretos sino que requiere una función global que mezcle todas las dimensiones, el FFN denso gana con un margen creciente (hasta 20x menos error) a medida que el presupuesto de parámetros se reduce.
2. **Más lento en cómputo real:** 30-56% más lento que un FFN a igualdad de parámetros en CPU, en todos los tamaños de batch probados — el ahorro teórico de FLOPs no compensa el costo de acceso disperso a memoria.
3. **El híbrido ingenuo (apilar bloques con presupuesto repartido) empeora, no mejora**, en las dos tareas de precisión y es el más lento en velocidad.
4. **Costo fijo no despreciable:** las proyecciones `W_q` y `W_gate` (cada una `d×d`) son parámetros "fijos" que no escalan con la capacidad de memoria — a dimensiones de modelo pequeñas, absorben una fracción grande del presupuesto total antes de que la memoria misma aporte nada.
5. **Desbalance de uso bajo datos sesgados** (aunque no until colapso total en mis pruebas — ver §7 sobre por qué esto podría empeorar a mayor escala).

## 7. Fallos teóricos esperables a gran escala (no probados aquí, pero documentados en la literatura de PKM/PEER)

| Riesgo | Por qué mi prueba no lo capturó | Por qué podría aparecer a escala real |
|---|---|---|
| **Colapso total de expertos** | Usé enrutamiento *suave* (softmax continuo sobre el top-k), que distribuye gradiente entre más claves que un top-k *duro* con straight-through estimator (lo que el texto original pide literalmente). | Con straight-through duro y datos de lenguaje real (mucho más sesgados que mi Zipf sintético), unas pocas claves pueden acaparar casi todo el tráfico desde las primeras iteraciones y nunca recuperarse ("rich get richer"), un problema bien documentado en Switch Transformer, PKM y PEER. |
| **Cuello de botella de ancho de banda de memoria en GPU** | Mi prueba corrió en CPU con tablas de 64 claves; no hay GPU ni tablas de millones de entradas. | Un `gather` sobre 1M+ entradas dispersas en HBM es memory-bound; sin kernel fusionado, la latencia real puede superar la de un FFN denso pese a tener menos FLOPs — ya lo vi incluso a micro-escala (§4.4). |
| **Inestabilidad del straight-through estimator** | No usé straight-through duro, sólo softmax continuo (más estable por diseño, pero también más costoso). | El gradiente sesgado del straight-through en top-k duro es conocido por producir entrenamientos ruidosos o divergentes sin ajuste cuidadoso de learning rate y temperatura. |
| **Interferencia en el compresor compartido** | Con pocos hechos y bottleneck pequeño no se saturó. | Si `bottleneck` es demasiado pequeño relativo al número de patrones distintos que deben pasar simultáneamente por él, se vuelve un segundo cuello de botella de capacidad, no solo de abstracción. |
| **Sharding/paralelismo distribuido** | Todo corrió en un solo proceso. | Repartir una tabla de `n_keys=1024` (o más) por lado entre múltiples GPUs exige comunicación all-to-all (como en MoE), añadiendo latencia y complejidad de ingeniería no reflejada en el conteo de FLOPs. |

---

## 8. Cómo mitigar cada fallo (sin destruir el diseño original)

1. **Colapso de expertos → pérdida auxiliar de balanceo de carga.** Añadir un término tipo Switch Transformer / PEER: `loss_total = loss_tarea + α · loss_balance`, donde `loss_balance` penaliza la varianza (o el coeficiente de variación) del uso de claves por batch. Coeficiente típico `α ≈ 0.01`; no requiere cambiar la arquitectura, solo el objetivo de entrenamiento.
2. **Claves muertas → reinicialización periódica.** Monitorear qué claves no se seleccionan en N pasos consecutivos y reinicializarlas (con ruido alrededor de la media de queries recientes) — técnica estándar en VQ-VAE y MoE para evitar que capacidad quede permanentemente desperdiciada.
3. **Cuello de banda ancha en GPU → kernel fusionado.** Implementar el `gather + low-rank matmul + suma ponderada` como un único kernel Triton/CUDA (en vez de operaciones separadas de PyTorch/JAX), similar a lo que hace FBGEMM para embeddings dispersos. Esto es exactamente lo que se necesitaría para que el ahorro teórico de FLOPs se refleje en latencia real.
4. **Inestabilidad del top-k duro → annealing suave-a-duro.** Empezar el entrenamiento con temperatura alta (blending suave sobre muchas combinaciones, como en mis pruebas) y bajarla gradualmente hacia un top-k casi duro al final, para quedarse con la eficiencia de inferencia dispersa sin pagar el costo de inestabilidad temprana.
5. **Costo fijo de `W_q`/`W_gate` → compartir o factorizar.** Atar estos pesos entre varias capas ECHO del modelo (weight tying), o reemplazarlos por proyecciones de bajo rango, para que el costo fijo no escale linealmente con el número de bloques ECHO en la red.
6. **Híbrido que diluye en vez de sumar → no repartir presupuesto de forma fija.** En vez de apilar ECHO y FFN secuencialmente con presupuesto partido a la mitad (lo que empeoró en mis pruebas), dos alternativas mejores:
   - **Mezcla aprendida por token:** compartir la misma entrada `h`, computar ambas ramas en paralelo (no en serie) y combinarlas con una compuerta escalar aprendida `g(h) ∈ [0,1]`: `out = g·ECHO(h) + (1-g)·FFN(h)`. El modelo decide dinámicamente cuánto usar de cada uno por token, en vez de forzar un orden fijo.
   - **Asignación asimétrica por capa:** en vez de mezclar dentro de cada capa, dedicar capas completas a cada tipo (p. ej. capas tempranas denso-FFN para composición sintáctica local, capas intermedias ECHO para recuperación de entidades/hechos), sin partir el presupuesto de ninguna capa individual.
7. **Interferencia en el cuello de botella compartido → escalar `bottleneck` con el número de combinaciones activas simultáneas**, no con `d` — mi regla práctica observada: mantenerlo en el orden de `k1·k2` (el número de combinaciones activas por token), no un valor fijo arbitrario.

Ninguna de estas mitigaciones cambia las tres reglas originales del diseño (end-to-end, una sola pasada, código simple) — son ajustes al objetivo de entrenamiento y a la ingeniería de kernels, no reescrituras de la arquitectura.

---

## 9. Cómo entrenar ECHO (recomendaciones prácticas)

- **Grupos de parámetros con learning rate distinto:** las tablas de claves (`K1`, `K2`) suelen beneficiarse de un LR distinto (a veces mayor) que las matrices densas del resto del bloque — práctica común en literatura de MoE/PKM.
- **Inicialización de claves a la escala de las queries:** inicializar `K1`/`K2` con la misma norma esperada que `q1`/`q2` tras el RMSNorm, para que el enrutamiento inicial no esté ya sesgado hacia unas pocas claves por pura escala numérica.
- **Monitorear entropía de uso como métrica de entrenamiento**, igual que hice en mis pruebas — es una señal barata de calcular y detecta desbalance mucho antes de que se vuelva colapso irreversible.
- **Pérdida de balanceo desde el primer paso** (no como parche posterior): añadirla desde el inicio es más barato que intentar corregir un colapso ya instalado.
- **Curriculum de temperatura/top-k** (blending suave → duro) si se necesita la eficiencia de inferencia dispersa; si no es crítica la latencia, el enrutamiento suave que usé en mis pruebas es más estable y ya mostró buenos resultados de capacidad.
- **Validar con datos realmente sesgados antes de escalar.** El experimento más informativo de todos los que corrí fue el de Zipf — recomendaría reproducirlo con la distribución de frecuencia real del corpus objetivo (no solo un Zipf sintético) antes de comprometer cómputo de GPU serio.

## 10. Pseudocódigo implementado (JAX, completando la parte que quedaba sin terminar)

```python
def echo_forward(p, h, k_topk):
    q = rmsnorm(h @ p["Wq"])
    d = h.shape[-1]
    q1, q2 = q[:, :d//2], q[:, d//2:]
    s1, s2 = q1 @ p["K1"].T, q2 @ p["K2"].T          # [B, n_keys] cada uno

    top1_val, top1_idx = jax.lax.top_k(s1, k_topk)
    top2_val, top2_idx = jax.lax.top_k(s2, k_topk)

    # softmax conjunta sobre las k*k combinaciones (equivalente continuo
    # del top-k con straight-through, diferenciable de punta a punta)
    combo_scores = top1_val[:, :, None] + top2_val[:, None, :]
    p_combo = jax.nn.softmax(combo_scores.reshape(B, -1), axis=-1).reshape(B, k, k)

    def expert_apply(down_table, up_table, idx, x):      # la parte que
        down = down_table[idx]                            # el texto original
        up = up_table[idx]                                # dejaba como
        z = jax.nn.silu(jnp.einsum("bd,bkdr->bkr", x, down))  # "# ... implementación
        return jnp.einsum("bkr,bkrd->bkd", z, up)             #  con gather ..."

    e1 = expert_apply(p["down1"], p["up1"], top1_idx, h)
    e2 = expert_apply(p["down2"], p["up2"], top2_idx, h)

    w1, w2 = p_combo.sum(axis=2), p_combo.sum(axis=1)     # pesos marginales
    m = jnp.einsum("bk,bkd->bd", w1, e1) + jnp.einsum("bk,bkd->bd", w2, e2)

    m_final = m + jax.nn.silu(m @ p["Wo1"]) @ p["Wo2"]
    return jax.nn.silu(h @ p["Wgate"]) * m_final
```

Bloque híbrido probado en §4.6 (composición secuencial con residual):
```python
def hybrid_forward(p, h, order):
    if order == "echo_first":
        h = h + echo_forward(p["echo"], h)
        h = h + ffn_forward(p["ffn"], h)
    else:
        h = h + ffn_forward(p["ffn"], h)
        h = h + echo_forward(p["echo"], h)
    return h
```

Compuerta aprendida recomendada en §8.6 (no probada aquí, propuesta de mitigación):
```python
def gated_hybrid_forward(p, h):
    e_out = echo_forward(p["echo"], h)
    f_out = ffn_forward(p["ffn"], h)
    g = jax.nn.sigmoid(h @ p["Wg"])          # compuerta escalar por token
    return g * e_out + (1 - g) * f_out
```

## 11. Cómo implementar a gran escala

1. **No reemplazar el 100% de los FFNs de entrada.** Mis propios resultados (§4.3, §4.6) muestran que ECHO pierde en tareas de mezcla suave y global de información, que es buena parte de lo que hace un FFN en un modelo de lenguaje real. Recomendación: reemplazar solo un subconjunto de capas (p. ej. 1 de cada 3-4), no todas.
2. **Kernel dedicado antes que nada.** Sin un kernel Triton/CUDA fusionado para el gather + low-rank matmul, cualquier ganancia teórica de FLOPs se pierde en la práctica (ya lo vi incluso a micro-escala en CPU, §4.4) — esto debería ser el primer ítem de ingeniería, no una optimización posterior.
3. **Pérdida de balanceo desde el día 1** (§8.1), no como parche.
4. **Sharding de las tablas de claves entre dispositivos** al estilo expert-parallelism de MoE si `n_keys` es grande (miles), con comunicación all-to-all — presupuestar esta latencia en cualquier estimación de throughput.
5. **Validar primero con la distribución de frecuencia real del corpus** (Zipf sintético como mínimo, corpus real como ideal) antes de comprometer cómputo — es la prueba que más señal dio en todo este proceso.
6. **Si se busca un híbrido, usar compuerta aprendida por token o asignación asimétrica por capa** (§8.6), no apilamiento secuencial con presupuesto partido — esto último empeoró en mis pruebas frente a cualquiera de los dos puros.
7. **Escalar el experimento en pasos, no de una vez:** 100-300M primero (validar estabilidad del entrenamiento con balanceo de carga y datos reales sesgados), luego 1-2B, antes de comprometer días de GPU en la escala que proponía el texto original.

## 12. Tabla resumen final

| Dimensión | Veredicto | Evidencia |
|---|---|---|
| Capacidad de memoria (datos sesgados, realista) | ✅ ECHO gana con margen amplio | §4.2: 15.2% vs 0.2% en cola rara |
| Capacidad de memoria (datos uniformes) | ✅ ECHO gana, margen moderado | §4.1: 64.3% vs 22.4% a 10K hechos |
| Regresión suave sin estructura de hechos | ❌ FFN gana, margen creciente | §4.3: 20x menos error a presupuesto reducido |
| Velocidad real | ❌ FFN gana consistentemente | §4.4: 30-56% más lento |
| Interpolación a puntos no vistos | ➖ Empate | §4.5 |
| Híbrido ingenuo (presupuesto partido) | ❌ Peor que ambos puros | §4.6 |
| Colapso de expertos (enrutamiento suave, datos sintéticos) | ⚠️ Parcial, no catastrófico | §4.2: entropía baja de 0.97 a 0.83-0.87 |
| Colapso con top-k duro + datos reales (no probado) | ⚠️ Riesgo teórico serio | §7, mitigable con §8.1-8.2 |
| Cifras del texto original (14-18B hechos, 2B>7B en razonamiento) | ❌ Sin sustento | No verificables a esta escala; siguen siendo extrapolación |

---

## 13. Conclusión

ECHO no es el "bloque perfecto sin compromisos" que planteaba el texto original, pero tampoco es una idea sin mérito. Es una especialización real con un trade-off honesto: gana de forma sustancial en memorización de datos sesgados (el caso que más se parece al lenguaje real), y pierde en velocidad real y en tareas de cómputo global suave. El camino razonable no es "reemplazar todo el FFN" ni "combinarlo todo intercalado" (esto último, probado aquí, empeora) — es usarlo selectivamente, con las mitigaciones de balanceo de carga y kernels dedicados descritas arriba, y validado primero contra datos con la distribución de frecuencia real del dominio objetivo antes de escalar.
