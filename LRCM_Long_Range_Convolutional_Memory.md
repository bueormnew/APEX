# LRCM — Long-Range Convolutional Retrieval Memory

**Especificación arquitectónica para un bloque de reemplazo de atención en modelos de lenguaje**  
**Estado:** diseño de investigación orientado a implementación  
**Objetivo:** sustituir la atención global cuadrática por mezcla de contexto de coste lineal/subcuadrático, manteniendo una ruta de recuperación exacta de información antigua y un camino local de atención normal.

> **Nota de rigor.** LRCM es una propuesta arquitectónica, no un resultado experimental demostrado. Las propiedades de coste y las decisiones de implementación de este documento son objetivos de diseño. En particular, “recuperación exacta” significa que, una vez localizado el fragmento correcto, la lectura final accede a los valores almacenados del fragmento sin depender exclusivamente de una representación comprimida. No significa equivalencia matemática con la salida de `softmax(QKᵀ)V` sobre todos los tokens del contexto.

---

## 1. Resumen ejecutivo

LRCM trata el problema de contexto largo como dos problemas distintos:

1. **Mezcla local:** relaciones de alta precisión entre tokens recientes.
2. **Recuperación de largo alcance:** encontrar y leer información que puede estar cientos de miles de tokens atrás.

La arquitectura no intenta calcular atención global barata. En su lugar, reemplaza la operación global por una **memoria jerárquica multiescala** que crece linealmente con el contexto y cuya tarea es localizar información. Una vez localizado un fragmento, LRCM hace una **lectura exacta sobre los tokens/valores almacenados de ese fragmento**.

La estructura conceptual es:

```text
                         LRCM BLOCK
                              │
                         input x_t
                              │
                         Pre-Norm
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       EXACT LOCAL MIXING          LONG-RANGE RETRIEVAL
       attention W=128..512               │
                │                 ┌───────┴────────┐
                │                 │                │
                │          multiscale summaries   │
                │          + long convolution     │
                │                 │                │
                │             hierarchical        │
                │              routing            │
                │                 │                │
                │            exact leaf read      │
                │                 │                │
                └────────────┬────┴────────────────┘
                             │
                       gated / normalized
                            merge
                             │
                       output projection
                             │
                          residual
```

La propiedad central es que **el contexto histórico completo deja de ser un conjunto de claves que cada token debe comparar**. Pasa a ser una estructura direccionable:

```text
1,048,576 tokens
        │
        ▼
4,096 chunks × 256 tokens
        │
        ▼
256 páginas × 4K tokens
        │
        ▼
16 regiones × 64K tokens
        │
        ▼
1 contexto global
```

La consulta baja por este árbol y termina en una hoja pequeña que puede leerse exactamente.

---

## 2. Motivación y requisitos

El objetivo no es demostrar que la atención sea innecesaria. El objetivo es diseñar un operador que tenga, para un LLM grande y especialmente para agentes, un comportamiento práctico más adecuado que la atención completa a contextos de 128K–1M.

### Requisitos de diseño

LRCM debe buscar simultáneamente:

- **Entrenamiento end-to-end:** los parámetros de mezcla local, memoria, direccionamiento y lectura deben poder recibir gradiente desde la pérdida causal.
- **Entrenamiento estable:** evitar que el modelo dependa de una fase obligatoria de conversión desde atención densa.
- **Interfaz de bloque convencional:** debe poder ocupar la posición de Attention dentro de un Transformer sin convertir toda la arquitectura en un sistema de anchors/followers.
- **Recuperación de largo alcance:** una variable, nombre, instrucción o fragmento de código situado a 128K o más debe poder localizarse y recuperarse.
- **Lectura exacta:** cuando la jerarquía decide dónde está la información, la última lectura debe usar la representación almacenada del fragmento real, no solamente su resumen.
- **Coste de mezcla histórico aproximadamente lineal:** el trabajo no debe crecer como `L²` con la longitud del contexto.
- **Memoria aproximadamente lineal:** permitir que el estado crezca con el contexto, pero mucho más lentamente que una matriz de interacciones completa.
- **Decode eficiente:** una consulta nueva debería leer sólo una cantidad pequeña y aproximadamente constante de candidatos históricos.
- **Kernel-friendly:** favorecer operaciones densas pequeñas, convoluciones/scan eficientes y accesos agrupados a memoria.
- **Versatilidad:** debe servir para lenguaje general, código, razonamiento y agentes.

### No objetivos

LRCM no pretende:

- reproducir exactamente los pesos de una atención global para todos los pares `(query, key)`;
- garantizar recuperación perfecta antes de tener datos experimentales;
- hacer toda la memoria completamente comprimida y destruir los valores originales;
- depender obligatoriamente de un patrón fijo de capas globales y locales.

---

## 3. Principio fundamental: localizar no es lo mismo que leer

Este es el principio más importante de LRCM.

Un sistema de recuperación de contexto largo tiene dos funciones diferentes:

### A. Addressing

Responder:

> “¿En qué parte del contexto está la información que necesito?”

Esta función puede trabajar con representaciones comprimidas.

### B. Recall

Responder:

> “Una vez localizado el lugar, ¿qué decía exactamente?”

Esta función debe acceder al contenido real del fragmento.

Separar estas dos operaciones evita exigir a una única representación comprimida que conserve simultáneamente todos los detalles y todas las relaciones globales.

---

## 4. Estructura multiescala de la memoria

La memoria histórica se organiza por bloques de posición fija.

Para una configuración de referencia de 1M de contexto:

| Nivel | Resolución | Cantidad aproximada |
|---|---:|---:|
| Hoja | 256 tokens | 4096 chunks |
| Página | 4K tokens | 256 páginas |
| Región | 64K tokens | 16 regiones |
| Global | 1M tokens | 1 raíz |

La relación es:

```text
1 región = 16 páginas
1 página  = 16 chunks
1 chunk   = 256 tokens
```

La aridad 16 no es una constante obligatoria. Puede probarse `8`, `16` o `32`. La razón para empezar en 16 es que permite búsquedas pequeñas y regulares.

La memoria lógica tiene forma de árbol:

```text
                         ROOT
                       1M tokens
                    /      ...      \
                  R0                R15
                64K each
              /   ...   \
            P0          P15
            4K each
           /  ...  \
         C0          C15
         256 each
```

No es necesario materializar un árbol de punteros. En GPU debe representarse como arrays contiguos por nivel para favorecer coalescencia y direccionamiento simple.

---

## 5. Representación de cada fragmento

Cada chunk conserva dos clases de información:

### 5.1. Contenido exacto

Para la capa o grupo de capas correspondiente se almacenan los valores necesarios para la lectura final.

Conceptualmente:

```text
LeafRecord = {
    key/value payload,
    positional metadata,
    validity mask
}
```

El formato físico puede ser FP16/BF16 en entrenamiento y FP8/INT8 u otro formato de inferencia apropiado, siempre que la degradación se mida explícitamente.

### 5.2. Descriptor de dirección

Cada chunk tiene un descriptor pequeño:

```text
ChunkDescriptor = {
    address_vector,
    content_summary,
    position_code,
    confidence
}
```

El descriptor no debe pretender almacenar todo el contenido. Su trabajo es permitir que una consulta encuentre el fragmento correcto.

Los niveles superiores se construyen sobre estos descriptores.

---

## 6. Construcción de la memoria con convolución larga

La intuición original de usar convoluciones largas es adecuada, pero hay que cambiarles el papel.

Una convolución larga **no debería ser la operación que hace la recuperación exacta**. Su función principal es construir una memoria multiescala y una representación de direccionamiento estable.

### Nivel hoja

Los tokens producen descriptors de chunk:

```text
X tokens
   │
   ├── local feature extraction
   └── chunk pooling / projection
          │
          ▼
      C_0 ... C_4095
```

### Nivel página

Los 16 chunks de una página se mezclan mediante una operación larga causal/gated:

```text
16 chunk descriptors
          │
          ▼
   long convolution / scan
          │
          ▼
     page descriptor
```

### Nivel región

Las 16 páginas producen un descriptor de región mediante el mismo principio.

### Nivel global

Las 16 regiones forman el resumen de contexto.

Por tanto:

```text
Tokens
  ↓
Chunk descriptors
  ↓ long-range convolution / gated scan
Page descriptors
  ↓ long-range convolution / gated scan
Region descriptors
  ↓ long-range convolution / gated scan
Global descriptor
```

La ventaja es que la convolución/scan opera sobre una secuencia de representaciones mucho más pequeña en cada nivel, en vez de crear una matriz token×token.

Hyena demostró que las convoluciones largas con control dependiente de datos pueden utilizarse como operadores de mezcla de contexto de gran longitud y mantener coste subcuadrático. Esa línea de trabajo sirve como precedente para la rama de memoria de LRCM, pero LRCM añade una ruta explícita de recuperación de hojas. [Hyena, ICML 2023](https://proceedings.mlr.press/v202/poli23a.html)

---

## 7. Consulta de memoria

La consulta global no debe provenir únicamente de `x_t`.

Primero se ejecuta la rama local:

```text
x_t
 │
 ▼
local attention
 │
 ▼
h_local
```

Después se construye:

\[
q_t^{mem} = W_q [x_t ; h_{local}]
\]

Esto es importante para un agente.

La consulta global debe representar el estado de trabajo actual, no solamente el token aislado.

Por ejemplo, si el modelo está generando:

```text
config.database_url = ...
```

la consulta puede representar implícitamente que está intentando completar una configuración y no simplemente que el último token pertenece a una cadena de texto.

---

## 8. Routing jerárquico

El router trabaja de grueso a fino.

Para una consulta `q`:

```text
q
 ↓
score 16 regiones
 ↓
select 1–4 regiones
 ↓
score páginas dentro de esas regiones
 ↓
select 1–4 páginas
 ↓
score chunks dentro de esas páginas
 ↓
select 1–4 chunks
 ↓
exact recall
```

### Complejidad de routing

Con fanout 16 y beam pequeño `B`:

```text
root/regions:        16
pages:               16 × B
chunks:              16 × B
```

Con `B=2`:

```text
16 + 32 + 32 = 80
```

comparaciones de descriptor antes de la lectura exacta.

No estamos comparando un query con 1M claves.

---

## 9. Cómo evitar que el routing destruya el entrenamiento

Este es el problema técnico más importante del sistema.

Un `argmax/top-k` duro no es una operación cómoda para optimización end-to-end porque corta el gradiente hacia las ramas descartadas.

La implementación propuesta tiene tres mecanismos compatibles.

### 9.1. Soft routing en cada nodo

En cada nivel se calcula:

\[
p_i = \operatorname{softmax}(q^T a_i / \tau)
\]

con una temperatura `τ` controlable.

### 9.2. Sparse beam con straight-through

Durante el entrenamiento se selecciona un beam pequeño:

```text
forward: top-B
backward: gradient through the soft probabilities
```

La selección discreta afecta el forward, pero los scores conservan una aproximación de gradiente.

No se debe empezar con temperaturas extremadamente bajas. La política inicial debe ser suave y posteriormente concentrarse.

### 9.3. Fallback de entrenamiento

En las primeras etapas puede evaluarse el routing sobre una cantidad pequeña de candidatos vecinos del camino elegido, en lugar de forzar una búsqueda completamente dura desde el primer paso.

El objetivo es evitar que el router colapse temprano a una región arbitraria.

> Esta parte debe considerarse un componente experimental de la implementación y medirse por separado. No debe presentarse como “resuelto” hasta obtener estabilidad reproducible.

---

## 10. Lectura exacta de la hoja

Una vez seleccionados los chunks, LRCM abandona la representación resumida.

Supongamos que el router encuentra:

```text
Region 2
Page 7
Chunk 11
```

El sistema obtiene los valores reales del chunk:

```text
V_exact[chunk]
```

y ejecuta una atención pequeña:

\[
O_{recall}
=
\operatorname{softmax}
\left(
\frac{QK_{exact}^{T}}{\sqrt d}
\right)V_{exact}
\]

Esto se hace sólo sobre los chunks recuperados.

Con 4 chunks × 256 tokens:

```text
1024 exact tokens
```

Por tanto, el modelo conserva una ruta para recuperar detalles exactos sin leer todo el millón.

---

## 11. ¿Qué significa “recuperación exacta” en LRCM?

Se deben distinguir tres niveles.

### Exactitud de almacenamiento

El contenido de una hoja recuperada es el contenido real almacenado para ese fragmento, no únicamente un resumen.

### Exactitud de lectura

La atención final sobre ese fragmento es atención normal sobre sus claves/valores.

### No equivalencia a atención global

El sistema no calcula:

\[
\operatorname{softmax}(QK_{1:L}^{T})V_{1:L}
\]

para todos los tokens.

Por eso no puede afirmar que produce exactamente la misma salida que full attention para cada posible entrada.

La garantía arquitectónica buscada es:

\[
\boxed{
\text{correctly locate} + \text{exactly read leaf}
}
\]

No:

\[
\boxed{
\text{replicate every global softmax interaction}
}
\]

---

## 12. Rama local

La rama local mantiene atención exacta con una ventana configurable:

```text
W = 128
W = 256
W = 512
```

Configuración inicial recomendada:

```text
W = 256
```

La operación es causal y convencional.

Esto proporciona una ruta robusta para información extremadamente cercana, donde sustituir atención por convolución o memoria puede ser innecesariamente arriesgado.

La local attention se encarga especialmente de:

- sintaxis inmediata;
- dependencias de corto alcance;
- operaciones de código consecutivas;
- copia local;
- correferencias cercanas;
- detalles recientes de una trayectoria de razonamiento.

---

## 13. Fusión de local y long-range

No se deben concatenar sin control las dos salidas.

Usar una mezcla aprendida:

\[
h = g_l \odot h_{local}
  + g_r \odot h_{recall}
  + g_m \odot h_{memory}
\]

con gates por grupo o por cabeza.

Una versión conservadora es:

\[
[g_l,g_r,g_m]
=
\operatorname{softmax}(W_g z)
\]

Esto garantiza una distribución controlada de contribuciones.

### Tres rutas de información

1. `h_local`: atención exacta inmediata.
2. `h_recall`: atención exacta sobre fragmentos antiguos recuperados.
3. `h_memory`: señal comprimida multiescala cuando la evidencia útil está distribuida o cuando la recuperación exacta no es suficiente.

No es obligatorio usar `h_memory` en la primera implementación. Puede eliminarse en la ablación inicial y quedar:

```text
local + exact retrieved chunks
```

---

## 14. ¿Para qué sirve la memoria comprimida si también existe exact recall?

Hay tres razones.

### 14.1. Información distribuida

Una respuesta puede depender de evidencia repartida en varias zonas del contexto.

La memoria multiescala puede representar una señal global antes de decidir dónde leer exactamente.

### 14.2. Routing

Los descriptores son el índice aprendido del sistema.

### 14.3. Contexto semántico global

La memoria puede capturar patrones de fondo que no corresponden a una única hoja.

Por tanto:

```text
Memory = “qué existe y dónde”
Exact recall = “qué contiene exactamente”
```

---

## 15. Memoria que crece linealmente

El objetivo no es memoria constante.

Para longitud `L`, el almacenamiento histórico es aproximadamente:

\[
O(Ld_{leaf}) + O((L/B)d_{chunk}) + O((L/BP)d_{page}) + O((L/BPG)d_{region})
\]

La primera parte domina porque contiene el contenido recuperable. Los niveles superiores son mucho más pequeños.

Para 1M y chunks de 256:

```text
4096 leaf descriptors
256 page descriptors
16 region descriptors
1 global descriptor
```

La cantidad de metadatos crece linealmente, pero los niveles superiores crecen lentamente.

---

## 16. Memoria física recomendada

Para producción no se debería asumir BF16 para absolutamente todo.

Separar:

```text
Local KV/cache:
    BF16 / FP16 / FP8 según backend

Exact historical payload:
    FP8 / INT8 candidato inicial

Descriptors:
    FP16/BF16 o FP8 si la calidad lo permite

Routing metadata:
    FP16 / FP8
```

El payload exacto debe permanecer suficientemente fiel para que la lectura sobre los chunks recuperados no pierda capacidad de copia y recuperación.

La cuantización debe evaluarse por separado de la arquitectura.

---

## 17. Compartición entre capas

Hay dos opciones.

### Opción A — memoria propia por capa

Cada bloque mantiene sus propios descriptors y leaf payload.

Ventajas:

- máxima expresividad;
- implementación conceptual sencilla;
- sin dependencia entre capas.

Desventajas:

- memoria histórica multiplicada por el número de capas.

### Opción B — memoria compartida por grupos de capas

Un pequeño grupo de capas utiliza un espacio de memoria compartido.

Por ejemplo:

```text
Layer 1 ─┐
Layer 2 ─┤
Layer 3 ─┤ → memory group A
Layer 4 ─┘

Layer 5 ─┐
Layer 6 ─┤
Layer 7 ─┤ → memory group B
Layer 8 ─┘
```

La representación consultada puede proyectarse a cada capa mediante una pequeña transformación.

Esta opción es más atractiva para producción porque reduce el multiplicador de almacenamiento, pero también aumenta la posibilidad de conflicto entre necesidades de capas diferentes.

**Configuración inicial:** memoria propia por bloque para el prototipo; compartir sólo después de medir la pérdida.

---

## 18. Exact leaf cache y estructura física

Una implementación GPU no debería utilizar un árbol de objetos.

Usar arrays contiguos:

```text
region_desc[B, R, D]
page_desc[B, P, D]
chunk_desc[B, C, D]
leaf_kv[B, C, T, ...]
```

con índices deterministas:

```text
region_id
page_id
chunk_id
```

La ventaja es que el router puede producir posiciones enteras y la lectura final puede transformarlas directamente en direcciones de memoria.

Para producción, el objetivo debe ser que los chunks candidatos queden almacenados de forma agrupada para minimizar transacciones dispersas.

---

## 19. El kernel ideal de decode

Para un token generado:

```text
1. load current hidden state
2. run local QKV
3. local attention over W
4. construct memory query
5. score R region descriptors
6. select beam
7. score child page descriptors
8. select beam
9. score child chunk descriptors
10. gather exact KV for selected chunks
11. run tiny exact attention
12. compute gates
13. merge
14. output projection
15. update memory/descriptors
```

No hay una multiplicación `Q × K_all`.

El coste dominante debe desplazarse a:

```text
local attention
small descriptor dot-products
small exact attention
```

---

## 20. Prefill

Durante el prefill todo el contexto está disponible.

Se aprovecha esto para construir la memoria de forma paralela:

```text
tokens
  ↓
chunk projections
  ↓
chunk descriptors
  ↓
page reductions / long conv
  ↓
region reductions / long conv
```

La construcción de niveles superiores debe realizarse con operaciones vectorizadas/chunkwise, no mediante un bucle Python por token.

Para long convolutions, utilizar una implementación compatible con FFT, scan o kernels fused dependiendo de la longitud y el hardware. No se debe asumir que la formulación asintótica por sí sola garantiza ventaja real: algunos modelos lineales recientes señalan que una operación teóricamente lineal puede resultar limitada por eficiencia de hardware. [Mamba-3, 2026](https://arxiv.org/abs/2603.15569)

---

## 21. Decode y actualización incremental

En decode no se recalcula toda la memoria histórica.

Sólo se actualiza el camino afectado por el nuevo token:

```text
new token
   ↓
current chunk descriptor
   ↓
page aggregate update
   ↓
region aggregate update
   ↓
root update
```

Una alternativa más eficiente es actualizar sólo cuando el chunk se completa y utilizar estados incrementales dentro de cada nivel.

La implementación debe soportar ambos modos:

```text
streaming update
chunk-finalized update
```

El segundo reduce escrituras de memoria.

---

## 22. Convolución larga: diseño concreto recomendado

La primera versión no debería intentar reproducir toda Hyena.

Usar una operación simple y estable:

```text
projection
 → depthwise causal long filter
 → gated mixing
 → projection
```

El filtro puede ser:

- parametrizado implícitamente;
- SSM-equivalent;
- o una convolución causal eficiente.

Una versión estilo Hyena es conceptualmente atractiva porque combina convolución larga con gating dependiente de datos. [Hyena, ICML 2023](https://proceedings.mlr.press/v202/poli23a.html)

Sin embargo, para la primera implementación, la prioridad debe ser un kernel mantenible y verificable, no maximizar sofisticación.

---

## 23. ¿Por qué no hacer simplemente una Hyena larga?

Porque una convolución larga puede propagar información a través de enormes distancias, pero la operación de mezcla sigue siendo una transformación comprimida.

Si una memoria histórica contiene:

```text
variable = 0x81F3...
```

y esa variable sólo aparece una vez, una transformación global podría necesitar conservar exactamente esa asociación para recuperarla posteriormente.

La memoria jerárquica de LRCM añade una segunda capa:

```text
long convolution → tells us where
exact leaf store → tells us what
```

Ese desacoplamiento está diseñado específicamente para tareas de recuperación.

---

## 24. Relación con modelos de memoria lineal

Los modelos recurrentes/lineales modernos demuestran que una memoria de tamaño fijo puede realizar mezcla de secuencia en tiempo lineal, pero el principal reto es cómo editar y recuperar información específica sin que las asociaciones interfieran entre sí.

Gated DeltaNet-2, por ejemplo, separa los mecanismos de borrado y escritura y conserva una formulación paralelizable por chunks. Sus resultados recientes muestran mejoras especialmente en escenarios de recuperación con interferencia. [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)

Kimi Linear utiliza KDA como núcleo de atención lineal y reporta reducciones importantes de KV cache y mejoras de throughput a contexto de 1M; su resultado también demuestra que el diseño de kernels y la arquitectura están estrechamente ligados. [Kimi Linear](https://arxiv.org/abs/2510.26692)

LRCM no sustituye esas líneas de investigación; las complementa con una memoria histórica direccionable de resolución múltiple.

---

## 25. Relación con memoria jerárquica

Titans separa conceptualmente memoria de corto plazo y memoria de largo plazo, y muestra que una memoria neuronal puede entrenarse de forma paralelizable mientras conserva capacidad para contexto muy largo. [Titans](https://arxiv.org/abs/2501.00663)

HMT utiliza memoria jerárquica a nivel de segmentos para mejorar la selección y filtrado de información histórica. [HMT, NAACL 2025](https://aclanthology.org/2025.naacl-long.410/)

La diferencia conceptual de LRCM es que la jerarquía está construida como un **índice multiescala sobre hojas exactas**, y la consulta puede terminar en una lectura atencional pequeña sobre el contenido real.

---

## 26. Estabilidad de entrenamiento

La arquitectura debe tener una ruta de optimización sencilla.

La formulación recomendada es pre-norm:

```python
x0 = x
z = rmsnorm(x)

h_local = LocalAttention(z)

q_mem = MemoryQuery(z, h_local)
h_mem, route = LongRangeMemory(q_mem, memory)
h_recall = ExactRecall(q_mem, route, memory)

h = Fuse(h_local, h_mem, h_recall)
y = x0 + OutputProjection(h)
```

La memoria debe actualizarse a partir de representaciones normalizadas y con gates acotados.

Evitar gates con escala libre al principio. `sigmoid` o `softmax` es preferible a una multiplicación sin normalización.

---

## 27. Inicialización recomendada

Para la primera versión:

```text
local branch:
    estándar del Transformer base

memory branch:
    salida inicial pequeña

recall branch:
    gate inicialmente bajo

output projection:
    inicialización estándar
```

Esto permite comenzar cerca del comportamiento de un bloque Transformer local y hacer que el camino de largo alcance aparezca de forma progresiva sin necesitar convertir un modelo ya entrenado.

No obstante, el gate bajo **no significa que el modelo deba congelarse**. Todas las ramas deben recibir gradiente desde el comienzo.

---

## 28. Objetivo de pérdida

La pérdida principal sigue siendo exactamente la habitual de un LLM causal:

\[
L_{LM}
=
-\sum_t \log p(x_t|x_{<t})
\]

No es necesario introducir una pérdida de atención artificial para que la arquitectura funcione.

### Pérdidas auxiliares opcionales

Se pueden añadir únicamente si los experimentos muestran un problema específico:

#### Recall consistency

Favorecer que una consulta que recupera correctamente una hoja asigne alta probabilidad al descriptor correspondiente.

#### Routing entropy / load balancing

Evitar colapso permanente del router a unas pocas ramas.

#### Memory reconstruction

Comprobar que el descriptor superior conserva suficiente información para distinguir fragmentos relevantes.

Estas pérdidas son **opcionales**. El benchmark principal debe ser un modelo que pueda entrenarse con `L_LM` solamente o casi solamente.

---

## 29. Entrenamiento por chunks

A contextos largos, el entrenamiento debe dividirse en chunks para reducir uso temporal de memoria.

Ejemplo:

```text
sequence = 1M
chunk = 4096

chunk 0 → local + memory state
chunk 1 → local + memory state
chunk 2 → local + memory state
...
```

Los estados de memoria se pasan entre chunks.

La contribución local permanece limitada a `W`.

La memoria histórica se mantiene en su estructura compacta.

La exact recall durante entrenamiento puede restringirse a una cantidad pequeña de hojas para controlar el coste.

---

## 30. Curriculum de contexto

No es necesario entrenar inicialmente siempre a 1M.

Una receta razonable es:

```text
4K
 ↓
16K
 ↓
32K
 ↓
64K
 ↓
128K
 ↓
256K
 ↓
512K
 ↓
1M
```

El operador es el mismo en todas las etapas. Sólo cambia el número de niveles activos.

Por ejemplo:

```text
4K   → chunk + page
64K  → chunk + page + region
1M   → chunk + page + region + global
```

Esto no es una “conversión” del modelo. Es una extensión del mismo bloque.

---

## 31. Problema crítico: router miss

Este es el riesgo principal.

Si el sistema decide:

```text
variable @ 128K
        ↓
wrong region
```

la lectura exacta no puede recuperar algo que nunca fue seleccionado.

Por eso el sistema debe maximizar `Recall@K` del router antes de optimizar la calidad de la atención final.

### Métricas obligatorias

```text
Region Recall@1
Region Recall@2
Page Recall@1
Page Recall@4
Chunk Recall@1
Chunk Recall@4
Exact retrieval accuracy
```

Una buena señal sería que la información relevante aparezca dentro de los pocos chunks seleccionados con alta frecuencia antes de aumentar la complejidad del lector.

---

## 32. Solución al router miss: beam pequeño

No usar sólo:

```text
Top-1 region
Top-1 page
Top-1 chunk
```

La configuración inicial debe ser:

```text
regions: top-2
pages:   top-2 per region
chunks:  top-2 per page
```

Esto genera hasta 8 hojas finales por consulta.

Con 256 tokens por hoja:

```text
8 × 256 = 2048 tokens exactos
```

Es más caro que 4 hojas, pero sigue siendo muy inferior a leer el contexto completo.

Después se puede estudiar:

```text
1 / 2 / 4 / 8 leaves
```

---

## 33. Solución adicional: ruta de escape

Para tareas extremadamente difíciles se puede reservar una pequeña ruta de escape.

Ejemplo conceptual:

```text
retrieval confidence < threshold
             │
             ▼
      expand beam / level
             │
             ▼
       extra exact chunks
```

No debe existir un mecanismo que ocasionalmente vuelva a atención completa sobre 1M tokens como comportamiento normal. El fallback debe seguir estando limitado.

Una alternativa de producción es permitir un presupuesto explícito:

```text
normal request: 4 chunks
hard request:   8 chunks
extreme:        16 chunks
```

El presupuesto se controla fuera del kernel mediante un único parámetro.

---

## 34. Problema crítico: información distribuida

No toda respuesta está contenida en un solo fragmento.

Ejemplo:

```text
Chunk A: nombre de variable
Chunk B: tipo
Chunk C: valor
Chunk D: regla que depende de ella
```

Solución:

- beam de múltiples hojas;
- representación global que resume contexto distribuido;
- posibilidad de hacer una segunda consulta después de la primera lectura.

Esta última opción es importante.

El modelo puede ejecutar:

```text
query 1
 ↓
retrieve A/B
 ↓
local reasoning
 ↓
query 2
 ↓
retrieve C
```

No es necesario resolver todo el contexto en una única operación de routing.

---

## 35. Problema crítico: información que cambia

En agentes, el contexto puede contener estados contradictorios:

```text
config = A
...
config = B
```

La memoria no debe sobrescribir automáticamente toda la historia.

Por eso cada hoja conserva posición y la consulta puede favorecer:

- relevancia semántica;
- proximidad temporal;
- consistencia con el estado actual.

La puntuación puede ser:

\[
s_i = q^T a_i + \alpha r_i + \beta p_i
\]

con términos separados para relevancia, recencia y posición.

No fijar `α` y `β` manualmente para siempre. Deben tratarse como parámetros o inputs aprendidos.

---

## 36. Problema crítico: deriva de los resúmenes

El descriptor de una región puede ir alejándose de la información que realmente contiene.

Por eso los niveles superiores deben ser **recalculables o actualizables** a partir de los niveles inferiores.

La memoria debería conservar:

```text
leaf payload
leaf descriptor
page descriptor
region descriptor
root descriptor
```

Esto permite regenerar niveles superiores si es necesario.

En producción, la implementación puede almacenar únicamente el mínimo necesario para el camino activo y reconstruir niveles superiores durante prefill.

---

## 37. Problema crítico: acceso irregular a memoria

La teoría puede decir “80 comparaciones”, pero una GPU puede tardar mucho si esas operaciones son accesos dispersos.

Por ello:

- descriptores contiguos por nivel;
- hojas agrupadas por página;
- beam limitado;
- gather fusionado;
- evitar estructuras dinámicas;
- tamaños de chunk fijos dentro de un kernel.

Para inferencia, el objetivo no es minimizar sólo FLOPs. Deben medirse:

```text
memory bandwidth
L2 hit rate
HBM traffic
kernel launch count
occupancy
latency per generated token
```

---

## 38. Problema crítico: memoria física

Aunque el crecimiento sea `O(L)`, un millón de tokens × muchas capas sigue pudiendo ser grande.

La solución no debe ser simplemente “guardar todo en BF16”.

La estrategia propuesta es:

```text
1. local KV:
   pequeño y rápido

2. long-term leaf payload:
   cuantizado / comprimido

3. descriptors:
   muy pequeños

4. upper levels:
   extremadamente pequeños

5. optional shared memory groups:
   para reducir multiplicación por capas
```

Esto convierte el problema de memoria en una cuestión de ingeniería controlable en vez de depender de una matriz `L × L` implícita.

---

## 39. Comparación conceptual de costes

Para longitud `L` y dimensión `d`:

### Full attention

Prefill de mezcla:

\[
O(L^2d)
\]

KV cache:

\[
O(Ld)
\]

por capa/grupo de KV.

### LRCM

Mezcla local:

\[
O(LWd)
\]

con `W` constante.

Memoria multiescala:

aproximadamente:

\[
O(Ld_{leaf})
\]

más descriptores de niveles superiores.

Routing por token:

\[
O(FB\log_F L)
\]

aproximadamente, donde `F` es la aridad del árbol y `B` el beam.

Exact recall:

\[
O(KRd)
\]

donde `K` es el número de hojas y `R` sus tokens exactos.

Con `K` y `R` pequeños y fijos, la lectura por token no escala linealmente con todo el contexto.

---

## 40. Una configuración concreta para 128K

```text
context      = 131072
chunk        = 256
page         = 4096
region       = 32768 o 65536
local        = 256
beam         = 4
exact leaves = 4
```

El árbol puede ser:

```text
128K
 ↓
2 × 64K
 ↓
16 × 4K dentro de cada región
 ↓
16 × 256 dentro de cada página
```

El objetivo es que una variable a 128K sea recuperable mediante una cadena de pequeñas decisiones.

---

## 41. Una configuración concreta para 1M

```text
context      = 1,048,576
chunk        = 256
page         = 4096
region       = 65536
local        = 256
region beam  = 2
page beam    = 2
chunk beam   = 2
```

Máximo aproximado de hojas:

```text
2 × 2 × 2 = 8
```

Lectura exacta máxima:

```text
8 × 256 = 2048 tokens
```

La memoria conserva toda la historia a resolución de hoja, pero una consulta individual sólo necesita visitar una fracción minúscula.

---

## 42. Arquitectura completa de referencia

```text
                    ┌───────────────────────────┐
                    │        INPUT x_t          │
                    └─────────────┬─────────────┘
                                  │
                               RMSNorm
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
                    ▼                            ▼
             LOCAL ATTENTION             MEMORY QUERY
              window = 256                     │
                    │                          │
                    │                   q = f(x, h_local)
                    │                          │
                    │                          ▼
                    │                   GLOBAL ROUTER
                    │                          │
                    │                ┌─────────┴─────────┐
                    │                ▼                   ▼
                    │           REGION SCORES        MEM SIGNAL
                    │                │
                    │             top-B
                    │                │
                    │             PAGE SCORES
                    │                │
                    │             top-B
                    │                │
                    │            CHUNK SCORES
                    │                │
                    │             top-B
                    │                │
                    │          EXACT LEAF GATHER
                    │                │
                    │                ▼
                    │          EXACT ATTENTION
                    │                │
                    └──────────┬─────┘
                               │
                         MEMORY SUMMARY
                               │
                         gated fusion
                               │
                         output projection
                               │
                            residual
```

---

## 43. Interfaz de implementación

El bloque debe exponer una API conceptualmente simple:

```python
class LRCMBlock(nn.Module):
    def forward(
        self,
        hidden_states,
        memory,
        position_ids=None,
        attention_mask=None,
        inference=False,
    ):
        ...
        return hidden_states, memory
```

La memoria debe ser una estructura explícita:

```python
class LRCMMemory:
    leaf_kv
    chunk_desc
    page_desc
    region_desc
    root_desc
    positions
```

En entrenamiento puede ser un contenedor de tensores.

En producción puede mapear directamente a buffers CUDA/FP8 y a páginas gestionadas por el runtime.

---

## 44. Pseudocódigo del forward

```python
z = norm(x)

# 1. Exact local path
h_local = local_attention(
    q_local(z),
    k_local(memory.local),
    v_local(memory.local),
    window=W,
)

# 2. Build retrieval query from current state + local evidence
q_mem = memory_query(z, h_local)

# 3. Hierarchical addressing
region_idx, region_score = route_regions(q_mem, memory.region_desc)
page_idx, page_score = route_pages(
    q_mem,
    memory.page_desc,
    region_idx,
)
chunk_idx, chunk_score = route_chunks(
    q_mem,
    memory.chunk_desc,
    page_idx,
)

# 4. Exact read from historical leaves
k_exact, v_exact = gather_leaf_kv(
    memory.leaf_kv,
    chunk_idx,
)

h_recall = exact_attention(
    q_mem,
    k_exact,
    v_exact,
)

# 5. Optional multiscale memory signal
h_mem = memory_read(
    q_mem,
    memory.region_desc,
    memory.page_desc,
    memory.chunk_desc,
)

# 6. Stable fusion
g = softmax(gate(torch.cat([h_local, h_mem, h_recall], dim=-1)))
h = g[..., 0:1] * h_local \
  + g[..., 1:2] * h_mem \
  + g[..., 2:3] * h_recall

# 7. Transformer residual
out = x + output_projection(h)

# 8. Update linear memory
memory = update_memory(memory, z, h_local, out)

return out, memory
```

La primera implementación puede eliminar `h_mem` y utilizar únicamente:

```text
local + exact recall
```

para reducir variables experimentales.

---

## 45. Primera versión que conviene construir

No comenzar con todos los mecanismos.

### LRCM-v0.1

```text
Local exact attention
+
256-token leaves
+
4K page descriptors
+
64K region descriptors
+
hierarchical routing
+
exact read
+
learned fusion
```

Sin:

```text
complex memory write
complex gating hierarchy
multiple attention heads in router
adaptive chunk sizes
```

Primero debe demostrar que:

```text
1M context
→ locate
→ exact read
```

funciona.

---

## 46. Ablaciones mínimas

El experimento no debe intentar validar veinte ideas a la vez.

### Ablación A — local solamente

```text
Local Attention W=256
```

### Ablación B — local + global convolution

```text
Local + long memory
```

### Ablación C — local + routing + compressed recall

Mide si el árbol realmente encuentra información.

### Ablación D — local + routing + exact leaf recall

Esta es la versión central.

### Ablación E — exact recall con diferentes beams

```text
B = 1 / 2 / 4 / 8
```

### Ablación F — memoria por capa vs grupos de capas

Mide coste/quality trade-off.

---

## 47. Tests de recuperación que deben existir antes del LLM grande

Antes de entrenar un modelo de miles de millones de parámetros, crear un benchmark artificial.

### Needle

Insertar:

```text
USER_ID = 728391
```

en posiciones aleatorias de 4K–1M.

Preguntar posteriormente:

```text
What is the USER_ID?
```

### Variable chaining

```text
A = 31
B = A * 7
C = B + 11
```

y solicitar `C` mucho después.

### Multiple variables

Insertar cientos de variables similares.

### Distractors

Repetir nombres casi idénticos:

```text
config_prod
config_production
config_prod_backup
```

Esto prueba si el router se limita a una coincidencia superficial.

### Long reasoning path

Separar la definición y el uso de una variable en 10K, 64K, 128K, 256K y 1M tokens.

---

## 48. Métricas principales

### Calidad de lenguaje

```text
Perplexity
validation loss
```

### Recuperación

```text
Needle-in-Haystack
RULER
multi-key retrieval
long-context QA
```

### Código

```text
variable retrieval
repository-level dependencies
long-file completion
cross-file reference retrieval
```

### Agentes

```text
tool state recall
instruction recall
multi-step state tracking
long trajectory memory
```

### Sistemas

```text
prefill tokens/s
prefill latency
decode tokens/s
single-token latency
peak VRAM
historical memory bytes/token
HBM traffic
```

---

## 49. Criterio de éxito

No basta con “funciona”.

La prueba fuerte debe ser:

```text
LRCM ≈ full attention en calidad de tareas relevantes
```

mientras que:

```text
LRCM << full attention en memoria y coste de mezcla global
```

Especialmente a:

```text
128K
256K
512K
1M
```

El resultado ideal sería un bloque que pierda muy poca calidad en recuperación pero convierta el problema práctico de contexto largo de:

```text
leer casi todo
```

en:

```text
local attention
+
pequeña búsqueda
+
pequeña lectura exacta
```

---

## 50. Producción: condiciones para considerarlo listo

LRCM no debería considerarse “production ready” sólo porque el modelo tenga buena perplexity.

Debe cumplir simultáneamente:

### Correctness

- causalidad correcta;
- no leakage de futuro;
- exact leaf indexing;
- positions correctas;
- determinismo configurable;
- recuperación estable bajo cuantización.

### Performance

- kernel fusionado;
- pocas kernel launches;
- gather eficiente;
- ausencia de Python en el camino crítico;
- memoria preasignada;
- batching compatible.

### Scaling

- 4K, 32K, 128K, 1M;
- diferentes batch sizes;
- diferentes head dimensions;
- múltiples GPUs si corresponde.

### Robustez

- datos con ruido;
- múltiples variables semejantes;
- contexto con código;
- historial de agente;
- documentos repetitivos.

---

## 51. Relación con trabajos existentes y posición de la propuesta

LRCM se inspira en varias familias, pero no debe atribuirse como una combinación completamente inédita sin una búsqueda de literatura y patentes.

### Hyena

Demostró una ruta de reemplazo de atención basada en convoluciones largas y gating, incluyendo tareas de recuperación y secuencias muy largas. [PMLR](https://proceedings.mlr.press/v202/poli23a.html)

### Titans

Separó conceptualmente memoria de corto y largo plazo y mostró una memoria neuronal capaz de trabajar con contextos superiores a 2M en sus experimentos. [arXiv](https://arxiv.org/abs/2501.00663)

### HMT

Usó memoria jerárquica y recurrencia a nivel de segmentos para procesamiento de contexto largo. [ACL Anthology](https://aclanthology.org/2025.naacl-long.410/)

### Gated DeltaNet-2

Mejoró la edición de memoria lineal separando borrado y escritura y manteniendo una ruta de entrenamiento por chunks. [arXiv](https://arxiv.org/abs/2605.22791)

### Mamba-3

Muestra que la eficiencia asintótica por sí sola no garantiza eficiencia de hardware y que la capacidad de state tracking sigue siendo una consideración central en modelos subcuadráticos. [arXiv](https://arxiv.org/abs/2603.15569)

### Kimi Linear

Demuestra que una arquitectura lineal/híbrida con kernels especializados puede escalar de forma muy favorable a 1M, con fuertes reducciones de KV cache y mejoras de throughput bajo su configuración. [arXiv](https://arxiv.org/abs/2510.26692)

La propuesta LRCM debe posicionarse como una arquitectura experimental que combina estas ideas en una estructura específica:

```text
local exact attention
+
multiscale long-convolutional descriptors
+
hierarchical learned routing
+
exact historical leaf recall
```

La novedad científica debe comprobarse; el valor principal en este documento es definir un mecanismo implementable que pueda ser falsado mediante experimentos claros.

---

## 52. Qué NO debe añadirse en la primera versión

Evitar introducir simultáneamente:

```text
pyramid attention
multi-stage Top-K attention
MQA
new positional encoding
QK normalization experimental
dynamic number of hierarchy levels
cross-layer arbitrary routing
latent memory writer complejo
learned token eviction
```

Cada uno añade una fuente adicional de inestabilidad o dificulta saber por qué el sistema funciona.

---

## 53. Extensiones posteriores

Una vez que la versión base funcione, hay varias direcciones naturales.

### Adaptive chunk size

Un documento homogéneo podría utilizar chunks grandes mientras que una zona con código podría dividirse con mayor resolución.

### Semantic reindexing

Permitir que una hoja pertenezca a varias claves semánticas sin copiar su contenido.

### Shared memory groups

Reducir memoria multiplicada entre capas.

### Multiple recall passes

Permitir que una recuperación genere una segunda consulta para encontrar evidencia relacionada.

### Learned compression levels

Hacer que las fronteras semánticas complementen las fronteras posiciones fijas.

### Hybrid recurrent global state

Añadir una memoria tipo DeltaNet para la señal global continua y utilizar LRCM para la recuperación explícita.

Esto último podría ser especialmente interesante:

```text
local attention
+
recurrent global state
+
long-convolution hierarchy
+
exact leaf recall
```

Pero debe considerarse una versión posterior, no parte de v0.1.

---

## 54. Variante avanzada: memoria lineal + LRCM

Una extensión potencialmente potente es colocar una memoria recurrente pequeña junto a la jerarquía.

```text
                 LONG RANGE
                    │
        ┌───────────┴───────────┐
        │                       │
 recurrent memory        hierarchical memory
   fast global signal       exact locator
        │                       │
        └───────────┬───────────┘
                    │
               exact recall
```

La memoria recurrente respondería:

> “qué información global parece importante”.

La jerarquía respondería:

> “en qué fragmento histórico está”.

La hoja exacta respondería:

> “cuál es el contenido preciso”.

Es una extensión prometedora, pero introducirla desde el principio elimina la capacidad de diagnosticar el sistema base.

---

## 55. Propiedad esencial para modelos agentes

El diseño está especialmente pensado para un contexto como:

```text
system instructions
↓
tools
↓
project state
↓
files
↓
previous tool calls
↓
observations
↓
reasoning
↓
new tool call
```

En ese tipo de flujo, no toda la historia necesita atención continua.

Normalmente se necesitan tres operaciones:

```text
1. saber qué está pasando ahora;
2. encontrar una pieza antigua relevante;
3. leer esa pieza con fidelidad.
```

LRCM asigna cada operación a una ruta distinta:

```text
now                 → local attention
where               → hierarchical memory
what exactly        → exact leaf recall
```

Esa separación es una de las razones principales para preferir la arquitectura frente a una simple convolución global.

---

## 56. Diseño recomendado final

La versión que debería implementarse primero queda así:

```text
LRCM-v1

┌──────────────────────────────────────────────────────┐
│                 PRE-NORM LRCM                        │
│                                                      │
│  Input                                               │
│    │                                                 │
│    ├── Local Exact Attention (W=256)                 │
│    │          │                                      │
│    │          └─────────┐                            │
│    │                    ▼                            │
│    │              Memory Query                      │
│    │                    │                            │
│    │           Hierarchical Router                  │
│    │            1M → 64K → 4K → 256                 │
│    │                    │                            │
│    │                 top-B                           │
│    │                    │                            │
│    │             Exact Leaf Gather                  │
│    │                    │                            │
│    │             Tiny Exact Attention               │
│    │                    │                            │
│    └────────────────────┤                            │
│                         ▼                            │
│                    Gated Fusion                     │
│                         │                            │
│                  Output Projection                   │
│                         │                            │
│                     Residual                         │
│                         │                            │
│                  Memory Update                      │
└──────────────────────────────────────────────────────┘
```

La memoria se construye como:

```text
Token representations
      ↓
256-token leaf descriptors
      ↓
long convolution / gated scan
      ↓
4K page descriptors
      ↓
long convolution / gated scan
      ↓
64K region descriptors
      ↓
root descriptor
```

---

## 57. Configuración inicial propuesta

| Parámetro | Valor inicial |
|---|---:|
| Local window | 256 |
| Leaf size | 256 |
| Page size | 4K |
| Region size | 64K |
| Region beam | 2 |
| Page beam | 2 |
| Leaf beam | 2 |
| Exact tokens/query | hasta 2048 |
| Descriptor dim | 128–256 |
| Router heads | 1–4 grupos |
| Memory dtype | BF16/FP8 según pruebas |
| Local attention | GQA estándar |
| Fusion | softmax gate |
| Positional encoding | RoPE estándar en local; posición explícita en descriptors |
| Training | causal LM end-to-end |

No son hiperparámetros “demostrados”. Son un punto de partida conservador.

---

## 58. Orden correcto de implementación

### Paso 1 — simulador CPU/GPU

Implementar:

```text
chunking
page aggregation
region aggregation
routing
exact gather
```

sin entrenar todavía un LLM.

### Paso 2 — benchmark artificial

Comprobar:

```text
Recall@1
Recall@2
Recall@4
```

a 4K–1M.

### Paso 3 — bloque PyTorch

Integrar:

```text
local attention
+
LRCM
```

### Paso 4 — tiny LM

Aproximadamente 50M–150M parámetros para depuración.

### Paso 5 — modelo 300M–1B

Comparar calidad y velocidad.

### Paso 6 — optimización Triton/CUDA

Sólo después de tener la arquitectura correcta.

### Paso 7 — contexto 128K+

Extender progresivamente.

### Paso 8 — 1M

Optimizar memoria y kernels después de validar recuperación.

---

## 59. Principio de desarrollo

La arquitectura debe validarse en este orden:

```text
CORRECTNESS
   ↓
RETRIEVAL
   ↓
TRAINABILITY
   ↓
QUALITY
   ↓
LATENCY
   ↓
MEMORY
   ↓
KERNEL OPTIMIZATION
```

No al revés.

Un operador que sea extremadamente rápido pero pierda una variable a 128K no cumple el objetivo principal.

---

## 60. Conclusión

LRCM no trata la atención global como algo que deba aproximarse directamente.

La reemplaza por una arquitectura de recuperación de dos escalas fundamentales:

```text
LOCAL
= interacción exacta reciente

LONG RANGE
= localizar información histórica

EXACT LEAF
= recuperar su contenido real
```

La convolución larga proporciona un mecanismo eficiente para construir representaciones históricas multiescala. La jerarquía reduce el espacio de búsqueda de millones de tokens a unas pocas decisiones pequeñas. La lectura final vuelve a la representación exacta del fragmento encontrado.

El objetivo estructural es transformar:

```text
query × 1,000,000 keys
```

en:

```text
query
  ↓
~16 regions
  ↓
~16 pages
  ↓
~16 chunks
  ↓
2–8 exact leaves
```

Con ello, la memoria puede crecer linealmente con el contexto mientras el número de elementos leídos por consulta permanece pequeño.

La hipótesis central que debe comprobarse experimentalmente es:

> **Una representación jerárquica construida con convolución larga puede aprender a direccionar con suficiente precisión el contenido histórico para que una lectura exacta sobre pocas hojas sustituya la mayor parte de la atención global, sin destruir la calidad de lenguaje ni la capacidad de recuperación de largo alcance.**

Si esta hipótesis funciona, LRCM puede convertirse en un verdadero bloque alternativo de `Attention` para modelos largos, no porque reproduzca cada interacción de softmax, sino porque cambia el problema de “atender a todo” por “localizar y recuperar exactamente lo que importa”.

---

## Referencias principales

1. Poli et al., **Hyena Hierarchy: Towards Larger Convolutional Language Models**, ICML 2023.  
   https://proceedings.mlr.press/v202/poli23a.html

2. Behrouz et al., **Titans: Learning to Memorize at Test Time**, 2024/2025.  
   https://arxiv.org/abs/2501.00663

3. He et al., **HMT: Hierarchical Memory Transformer for Efficient Long Context Language Processing**, NAACL 2025.  
   https://aclanthology.org/2025.naacl-long.410/

4. Hatamizadeh et al., **Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention**, 2026.  
   https://arxiv.org/abs/2605.22791

5. Lahoti et al., **Mamba-3: Improved Sequence Modeling using State Space Principles**, 2026.  
   https://arxiv.org/abs/2603.15569

6. Kimi Team et al., **Kimi Linear: An Expressive, Efficient Attention Architecture**, 2025/2026.  
   https://arxiv.org/abs/2510.26692

---

## Estado de las afirmaciones

- **Implementable:** sí, a nivel de arquitectura y pseudocódigo.
- **End-to-end:** sí, como objetivo de diseño; la estabilidad del router debe verificarse experimentalmente.
- **Complejidad subcuadrática:** sí, por construcción; el coste exacto depende de la implementación de la convolución, router y leaf gather.
- **Recuperación exacta de hojas:** sí, siempre que el router seleccione la hoja correcta y el payload exacto esté disponible.
- **Equivalencia con full attention:** no.
- **Calidad comparable a full attention a 128K–1M:** todavía es una hipótesis experimental.
- **Novedad científica:** no declarada; requiere revisión de literatura y patentes antes de publicarse como novedad.
- **Producción:** objetivo de diseño, no afirmación de que el bloque ya esté validado en producción.
