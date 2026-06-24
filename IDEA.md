# IDEA.md — Visual Chain of Thought (V-CoT)

> **Una representación explícita del pensamiento visual.**
> No entrenar un modelo que mapea `Prompt → Imagen`, sino capturar y destilar la
> *secuencia de decisiones* que un humano experto toma antes de que exista un solo
> píxel — y medir, con rigor de investigación, el coste de inferencia de hacerlo
> sobre infraestructura serverless (Modal.com).

---

## 0. TL;DR

- **Hipótesis:** la inteligencia visual no vive en la imagen final, sino en la
  cadena de decisiones (intención → concepto → layout → composición →
  iluminación → materiales → color → render). Los modelos de difusión actuales
  colapsan toda esa cadena en una sola operación opaca.
- **Propuesta:** *Visual Chain of Thought* (V-CoT), el análogo visual del
  Chain-of-Thought textual. Cada paso intermedio es **observable**, **serializable**
  y **criticable**.
- **Plan experimental:**
  1. Usar un modelo grande (FLUX.1-dev como renderer + un LLM/VLM como *planner*)
     para generar un **dataset sintético de razonamiento visual** a gran escala.
  2. **Destilar** ese proceso en un modelo pequeño ("Klein", 4B–9B) que aprende a
     *pensar* visualmente, no solo a renderizar.
  3. **Medirlo todo**: latencia por etapa, throughput, memoria GPU, tokens/s,
     calidad — y sobre todo **coste por muestra y por capacidad** en Modal.
- **Infra:** Modal.com (serverless GPU). Cada etapa del pipeline es una `Function`
  independiente, lo que permite atribuir coste **por etapa** y elegir el GPU
  óptimo para cada una.
- **Entregable central:** no "mejores imágenes", sino un **Visual Reasoning Model**
  y un **modelo de costes de inferencia** reproducible.

---

## 1. Motivación y marco teórico

### 1.1 El problema

Un director de fotografía, un artista conceptual, un diseñador publicitario o un
arquitecto **nunca** saltan directamente a la imagen final. Existe una secuencia
cognitiva:

```text
Intención → Concepto → Layout → Composición → Iluminación → Materiales → Color → Refinamiento → Resultado
```

Los modelos de difusión texto-a-imagen actuales aprenden `Prompt → Imagen` y
colapsan toda esa cadena en un único paso. La "inteligencia" queda implícita y
no observable dentro de los pesos.

### 1.2 La hipótesis central

> Si capturamos la **secuencia de decisiones** que produce una imagen, un modelo
> mucho más pequeño podrá aprender capacidades visuales desproporcionadamente
> superiores — porque aprende **el proceso**, no solo el producto.

Es el mismo fenómeno que ya ocurrió en el dominio textual:

```text
GPT-4  →  trazas sintéticas de razonamiento  →  modelos pequeños con razonamiento fuerte
```

El equivalente visual:

```text
FLUX + Planner  →  trazas sintéticas de razonamiento visual (V-CoT)  →  Klein (4B–9B)
```

### 1.3 Por qué es publicable

- Define un **artefacto nuevo**: dataset de *pensamiento visual* (no de imágenes).
- Propone una **descomposición canónica** del proceso de creación visual en etapas
  observables y evaluables independientemente.
- Conecta tres líneas activas: difusión controlable, destilación de razonamiento,
  y *mixture-of-experts*.
- Aporta una **metodología de medición de inferencia y coste** reproducible sobre
  serverless — algo de lo que la literatura suele carecer.

---

## 2. El pipeline V-CoT

En lugar de `Prompt → Imagen`, el sistema produce una traza completa:

```text
Prompt
 ↓
[N1] Semantic Scene Plan      (semántica pura, sin píxeles)
 ↓
[N2] Spatial Layout           (scene graph + bounding boxes)
 ↓
[N3] Composition              (decisiones cinematográficas)
 ↓
[N4] Lighting Design          (esquema de luz)
 ↓
[N5] Material Definition      (materiales / superficies)
 ↓
[N6] Color Script             (paleta y temperatura)
 ↓
[N7] Final Render             (imagen)
```

Cada etapa es **texto/JSON estructurado** salvo N7 (y, opcionalmente, renders
intermedios de baja resolución para N2–N6). Esto hace toda la cadena auditable.

### 2.1 Esquema de cada etapa

**N1 — Semantic Scene Plan** (storyboard mental, sin píxeles):

```json
{
  "subject": "astronaut",
  "environment": "abandoned cathedral",
  "camera": "wide angle",
  "mood": "mysterious",
  "dominant_elements": ["stained glass", "fog", "moonlight"]
}
```

**N2 — Spatial Layout** (scene graph + geometría aproximada):

```json
{
  "canvas": [1024, 1024],
  "entities": [
    {"id": "astronaut",  "bbox": [0.35, 0.45, 0.65, 0.95], "z": 1},
    {"id": "moonlight",  "bbox": [0.00, 0.00, 0.40, 0.30], "z": 3},
    {"id": "cathedral",  "bbox": [0.00, 0.00, 1.00, 1.00], "z": 4},
    {"id": "fog",        "bbox": [0.00, 0.70, 1.00, 1.00], "z": 0}
  ]
}
```

**N3 — Composition**:

```json
{ "lens": "35mm", "rule_of_thirds": true, "subject_scale": 0.4,
  "leading_lines": true, "symmetry": false }
```

**N4 — Lighting**:

```json
{ "key_light": "moonlight", "fill_light": "low", "rim_light": true,
  "contrast": "high" }
```

**N5 — Materials**:

```json
{ "glass": "wet reflective", "stone": "aged gothic", "metal": "oxidized" }
```

**N6 — Color Script** (inspirado en pipelines de Pixar/Disney):

```json
{ "primary_palette": ["#0F172A", "#3B82F6", "#CBD5E1"],
  "temperature": "cold", "saturation": "medium" }
```

**N7 — Final Render:** imagen generada condicionada por la traza acumulada.

### 2.2 Visual Tokens (representación compacta)

Para entrenamiento eficiente, la traza se puede comprimir a una secuencia de
tokens discretos de razonamiento visual:

```text
PLAN_SUBJ:astronaut  ENV:cathedral  LAYOUT_C:center  LENS_35MM
LIGHT_KEY:moon  CONTRAST_HIGH  COLOR_COLD  SAT_MED  RENDER
```

Esto convierte la imagen en una **secuencia de razonamiento** y habilita
arquitecturas autoregresivas / MoE sobre los pasos.

---

## 3. Cómo se condiciona el render (N7)

El reto técnico es que las etapas N2–N6 **realmente influyan** en N7 (que no sean
decoración). Tres mecanismos, de menos a más fuerte:

1. **Prompt enrichment (baseline):** serializar la traza a un prompt enriquecido
   y pasarlo a FLUX. Barato, débil acoplamiento. *Buen punto de partida y control.*
2. **Conditioning espacial:** convertir N2 (layout) en mapas de control
   (bounding boxes → depth/canny/regional prompting) vía ControlNet / regional
   attention. Acoplamiento medio-fuerte de la geometría.
3. **Renders intermedios encadenados:** generar imágenes de baja resolución para
   N2–N4 (blocking, lighting pass) y usarlas como `init_image` / control de las
   siguientes etapas (img2img encadenado). Acoplamiento fuerte, **coste mayor**
   (clave para el análisis económico).

> **Decisión de diseño explícita:** el experimento mide los tres mecanismos como
> *condiciones* y reporta la curva **coste ↔ calidad ↔ controlabilidad**.

---

## 4. Generación del dataset sintético

### 4.1 Roles

- **Planner (N1–N6):** un LLM/VLM produce la traza estructurada. Dos opciones a
  comparar:
  - **(A) API externa** (p.ej. Claude) — alta calidad, coste por token externo,
    no consume GPU de Modal.
  - **(B) LLM self-hosted en Modal** (p.ej. un 7B–14B en vLLM) — coste en GPU de
    Modal, control total, reproducible offline.
- **Renderer (N7):** FLUX.1-dev en GPU de Modal.
- **Critic (opcional, ver §7):** VLM que evalúa cada etapa y permite
  auto-corrección.

### 4.2 Registro por muestra

Cada muestra del dataset se persiste como un único registro:

```json
{
  "id": "uuid",
  "prompt": "...",
  "semantic_plan": { ... },
  "layout": { ... },
  "composition": { ... },
  "lighting": { ... },
  "materials": { ... },
  "color_script": { ... },
  "visual_tokens": ["...", "..."],
  "final_image": "vol://images/uuid.webp",
  "intermediate_renders": ["vol://inter/uuid_n2.webp", "..."],
  "meta": {
    "planner": "claude|vllm-qwen2.5-14b",
    "renderer": "flux.1-dev",
    "render_steps": 28,
    "resolution": [1024, 1024],
    "seed": 12345
  },
  "telemetry": { /* ver §6: latencias y coste por etapa */ }
}
```

> El valor del dataset no son las imágenes — es la **estructura de decisiones**.
> `ImageNet = imágenes`. `V-CoT Dataset = decisiones visuales.`

### 4.3 Escalas objetivo

| Fase | Muestras | Propósito |
|------|----------|-----------|
| Smoke | 100 | Validar pipeline, instrumentación y coste real medido |
| Pilot | 10 000 | Curvas coste/calidad, elegir GPU por etapa |
| Alpha | 250 000 | Suficiente para destilación inicial de Klein |
| Beta | 1 000 000+ | Dataset de referencia |

---

## 5. Destilación, MoE y modelo objetivo (Klein)

### 5.1 Destilación

En lugar de entrenar `Prompt → Imagen`, Klein se entrena sobre la traza completa:

```text
Prompt → Semantic Plan → Layout → Composition → Lighting → Materials → Color → Image
```

Klein aprende a **generar la traza** y luego a renderizar condicionado por ella.
La hipótesis: a igualdad de parámetros, un modelo que produce el proceso supera
a uno que solo produce el producto.

### 5.2 Mixture of Visual Experts (MoVE)

Cada etapa puede tener un experto especializado:

```text
Planner Expert · Composition Expert · Lighting Expert · Material Expert · Color Expert · Render Expert
```

Esto encaja con un MoE donde el "router" es la **etapa del pipeline**. Ventaja
operativa en Modal: cada experto = una `Function` con su propio GPU y su propia
contabilidad de coste.

### 5.3 Self-correction (RL-style)

```text
Klein → Layout      Critic → "subject too small"   → Klein corrige
Klein → Lighting    Critic → "contrast too low"     → Klein corrige
```

Bucle iterativo tipo RLHF/RLAIF sobre etapas individuales. **Medible**: cuántas
iteraciones de corrección, cuánto coste extra por iteración, cuánta mejora de
calidad.

---

## 6. Instrumentación de inferencia (nivel researcher)

Objetivo: capturar **toda** la información de inferencia posible, por etapa y
extremo a extremo. Todo se vuelca a la telemetría de cada muestra (§4.2) y a un
parquet agregado en un Volume.

### 6.1 Métricas de latencia

Para **cada etapa** y para el pipeline completo:

- `cold_start_s` — tiempo de arranque del contenedor (crítico en serverless).
- `model_load_s` — carga de pesos a GPU (mitigable con `@enter` + snapshots).
- `queue_wait_s` — espera por disponibilidad de worker.
- `compute_s` — tiempo de cómputo puro (lo que se factura como GPU activo).
- `e2e_latency_s` — extremo a extremo percibido.
- Distribución completa: **p50 / p90 / p95 / p99**, no solo la media.

### 6.2 Métricas de throughput y recursos

- `images_per_hour`, `samples_per_hour` (con y sin batching).
- `tokens_per_s` (planner): prefill vs decode separados.
- `gpu_mem_peak_gb`, `gpu_util_pct` (vía `nvidia-smi` / `pynvml`).
- `steps_per_s` (renderer: pasos de difusión por segundo).
- `batch_size` efectivo y su efecto en coste por muestra.

### 6.3 Métricas de calidad (para cruzar con coste)

- Render: CLIP-score, FID/KID contra referencia, aesthetic predictor.
- Adherencia espacial: IoU entre N2 (layout pedido) y detección sobre N7.
- Adherencia de color: distancia en espacio Lab entre N6 y paleta extraída de N7.
- Coherencia de la traza: validez de esquema JSON, consistencia entre etapas.

### 6.4 Métricas de coste (el corazón del proyecto) — ver §8.

---

## 7. Arquitectura en Modal.com

### 7.1 Principio de diseño

> **Una `Function` por etapa.** Cada etapa elige su GPU óptimo y se factura por
> separado, lo que permite atribuir coste **por capacidad cognitiva** y no solo
> por imagen.

```text
app = modal.App("vcot")

@app.function(gpu=None,           ...)  planner_api        # N1–N6 vía API (sin GPU Modal)
@app.function(gpu="L4",           ...)  planner_vllm_small # N1–N6 self-hosted ligero
@app.function(gpu="A100-40GB",    ...)  planner_vllm_big   # N1–N6 self-hosted potente
@app.function(gpu="H100",         ...)  renderer_flux      # N7 (y renders intermedios)
@app.function(gpu="L40S",         ...)  critic_vlm         # self-correction
@app.function(gpu=None, cpu=...,  ...)  aggregator         # telemetría → parquet
```

### 7.2 Patrones clave de Modal (impacto directo en coste)

- **`@app.cls` + `@enter`:** cargar pesos una vez por contenedor, no por llamada.
  Reduce `model_load_s` de cada muestra.
- **`min_containers` / `keep_warm`:** evitar cold starts a cambio de coste de
  contenedor ocioso. **Trade-off central a medir.**
- **`max_containers` / autoscaling:** escalar fan-out para la generación masiva.
- **`.map()` / `.starmap()`:** paralelizar la generación del dataset (miles de
  prompts en paralelo).
- **Memory Snapshots / GPU snapshots:** reducir cold start cargando estado.
- **`modal.Volume`:** almacenar imágenes, renders intermedios y el dataset
  (1 TiB/mes gratis; luego $0.09/GiB/mes).
- **`Secret`:** API keys del planner externo.
- **Batching dinámico** (`@batched`) en el renderer para subir throughput/GPU.

### 7.3 Cronómetro de coste embebido

Un decorador/contexto propio mide `compute_s` real y lo multiplica por la tarifa
del GPU (§8.1) para obtener `cost_usd` por etapa **en cada llamada**, además de
reconciliarlo con la facturación real de Modal al final.

```python
with cost_timer(gpu="H100") as t:
    image = run_flux(...)
telemetry["render"] = {"compute_s": t.seconds, "cost_usd": t.cost}
```

---

## 8. Modelo de costes

> Tarifas de Modal proporcionadas (por segundo y por hora). El proyecto usa la
> **tarifa por segundo** como unidad base porque la facturación es por uso real.

### 8.1 Tabla de tarifas (referencia)

| Recurso | $/segundo | $/hora |
|---|---:|---:|
| GPU Nvidia B200 | 0.001736 | 6.25 |
| GPU Nvidia H200 | 0.001261 | 4.54 |
| GPU Nvidia H100 | 0.001097 | 3.95 |
| GPU Nvidia RTX PRO 6000 | 0.000842 | 3.03 |
| GPU Nvidia A100 80 GB | 0.000694 | 2.50 |
| GPU Nvidia A100 40 GB | 0.000583 | 2.10 |
| GPU Nvidia L40S | 0.000542 | 1.95 |
| GPU Nvidia A10 | 0.000306 | 1.10 |
| GPU Nvidia L4 | 0.000222 | 0.80 |
| GPU Nvidia T4 | 0.000164 | 0.59 |
| CPU (core físico ≈ 2 vCPU) | 0.0000131 / core | 0.0473 / core |
| Memoria | 0.00000222 / GiB | 0.0080 / GiB |
| Volumes | — | 0.09 / GiB / **mes** (1 TiB/mes gratis) |

> Mínimo 0.125 cores por contenedor. La memoria se factura aparte del GPU.
> El **cold start y el `model_load` también se facturan** (el contenedor está
> vivo): por eso son métricas de coste, no solo de latencia.

### 8.2 Fórmula de coste por muestra

```text
cost_sample =
    Σ_etapas [ (compute_s + load_s_amortizado) × (rate_gpu + rate_mem×GiB + rate_cpu×cores) ]
  + cost_planner_externo (si aplica)
  + cost_almacenamiento_amortizado
```

donde `load_s_amortizado = model_load_s / muestras_por_contenedor` (clave: con
`@enter` y contenedores cálidos, este término tiende a cero a escala).

`compute_s` total facturado por etapa incluye overhead de contenedor; el cómputo
puro se reporta aparte para análisis.

### 8.3 Estimaciones *a priori* (a sustituir por mediciones reales)

> Estos números son **placeholders de ingeniería** para dimensionar el
> presupuesto. El objetivo de la fase Smoke/Pilot es reemplazarlos por medidas
> reales. Todos suponen contenedores cálidos (`load_s` amortizado ≈ 0).

**Renderer N7 — FLUX.1-dev, 1024², 28 pasos:**

| GPU | compute_s (est.) | $/imagen | $/1 000 | $/1 000 000 |
|---|---:|---:|---:|---:|
| H100 | 6 s | 0.00658 | 6.58 | 6 582 |
| A100 80GB | 10 s | 0.00694 | 6.94 | 6 940 |
| L40S | 14 s | 0.00759 | 7.59 | 7 588 |
| B200 | 3.5 s | 0.00608 | 6.08 | 6 076 |

> Nota: el más caro por segundo (B200) puede ser el **más barato por imagen** si
> acelera lo suficiente. Esa es exactamente la métrica que el proyecto mide:
> **coste por unidad de trabajo, no por hora.**

**Planner N1–N6 — self-hosted (Qwen2.5-14B en vLLM, A100-40GB):**
suponiendo ~1 200 tokens de salida total para las 6 etapas a ~80 tok/s ⇒ ~15 s.

| GPU | compute_s (est.) | $/traza | $/1 000 |
|---|---:|---:|---:|
| A100 40GB | 15 s | 0.00875 | 8.75 |
| L40S | 18 s | 0.00976 | 9.76 |
| L4 | 40 s | 0.00888 | 8.88 |

**Planner N1–N6 — API externa (orden de magnitud):** ~1 200 tokens out + ~400 in.
Coste dependiente del proveedor/modelo; se mide aparte y se compara contra
self-hosted para decidir la opción más barata a cada escala.

**Coste por muestra completa (estimación, planner self-hosted + render H100):**

```text
≈ 0.00875 (planner)  +  0.00658 (render)  ≈ 0.0153 $/muestra
```

| Escala | Coste estimado (sin renders intermedios) |
|---|---:|
| Smoke (100) | ~$1.5 |
| Pilot (10 000) | ~$153 |
| Alpha (250 000) | ~$3 825 |
| Beta (1 000 000) | ~$15 300 |

> Con **renders intermedios** (mecanismo de conditioning fuerte, §3.3) el coste de
> render se multiplica por ~(1 + nº de pasos intermedios). Esa es una palanca de
> coste de primer orden que el estudio cuantifica explícitamente.

### 8.4 Coste de almacenamiento

```text
1 imagen WebP 1024² ≈ 150–300 KB.
1 000 000 imágenes finales ≈ 150–300 GB  ⇒  cabe en la franja gratuita (1 TiB/mes).
+ renders intermedios (×4–5) podría superar 1 TiB ⇒ excedente a $0.09/GiB/mes.
```

### 8.5 Cold start: coste oculto

```text
cold_start_overhead_$ = (cold_start_s + model_load_s) × rate_gpu
```

Ejemplo H100, cold 20 s + load 25 s = 45 s ⇒ **$0.049 por contenedor frío**.
Decisión a optimizar: ¿pagar `keep_warm` (contenedor ocioso facturado) o pagar
cold starts repetidos? El estudio traza esa frontera en función del *arrival rate*.

---

## 9. Experimentos

### E1 — Coste/calidad por GPU (renderer)
Renderizar el mismo set de prompts en {T4, L4, A10, L40S, A100-40, A100-80, H100,
H200, B200}. Reportar $/imagen vs FID/CLIP/aesthetic y la frontera de Pareto.
**Pregunta:** ¿cuál es el GPU óptimo *por dólar de calidad*?

### E2 — Mecanismo de conditioning (§3)
Comparar prompt-enrichment vs control espacial vs renders intermedios encadenados.
Reportar controlabilidad (IoU layout, distancia de color) vs coste.

### E3 — Planner: API vs self-hosted
Curva de coste cruzado: ¿a partir de cuántas muestras conviene self-hostear?
Reportar también diferencia de calidad de la traza.

### E4 — Cold start vs keep_warm
Barrer `min_containers` ∈ {0, 1, 2, ...} bajo distintos *arrival rates*. Reportar
p99 de latencia y $/muestra. Encontrar el punto de equilibrio.

### E5 — Batching del renderer
Barrer batch ∈ {1, 2, 4, 8}. Reportar throughput, GPU util y $/imagen.

### E6 — Destilación (la prueba de la hipótesis)
Entrenar Klein (4B–9B) sobre el dataset V-CoT vs un baseline `Prompt→Imagen` del
mismo tamaño. **Métrica clave:** ¿el modelo que aprende el *proceso* gana en
controlabilidad/calidad a paridad de parámetros y de coste de inferencia?

### E7 — Self-correction
Medir mejora de calidad por iteración de Critic vs coste extra por iteración.

---

## 10. Estructura del repositorio (propuesta)

```text
vcot/
  modal_app/
    app.py              # definición de la App y Functions
    planner.py          # N1–N6 (API + vLLM)
    renderer.py         # N7 FLUX + conditioning (3 mecanismos)
    critic.py           # self-correction
    schemas.py          # esquemas JSON/pydantic de cada etapa
    visual_tokens.py    # traza ↔ tokens
  telemetry/
    cost_timer.py       # cronómetro + cálculo de coste por etapa
    rates.py            # tabla de tarifas (§8.1), única fuente de verdad
    aggregate.py        # telemetría → parquet
  dataset/
    generate.py         # fan-out .map() para generación masiva
    record.py           # registro por muestra (§4.2)
  analysis/
    cost_report.ipynb   # fronteras de Pareto coste/calidad
    latency_report.ipynb
  train/
    distill_klein.py    # destilación
  IDEA.md               # este documento
```

> El proyecto actual (`src/inferencetest/`) sirve de scaffold; `telemetry/rates.py`
> debe ser la **única fuente de verdad** de las tarifas para que todo informe de
> coste sea reproducible y actualizable en un solo sitio.

---

## 11. Roadmap

| Hito | Contenido | Criterio de éxito |
|---|---|---|
| **M0 — Setup** | App Modal, Volume, Secret, `cost_timer`, `rates.py` | una imagen generada con coste real medido |
| **M1 — Pipeline V-CoT** | N1–N7 funcionando, registro completo por muestra | 100 muestras (Smoke) con telemetría completa |
| **M2 — Medición** | E1, E4, E5 | fronteras de Pareto coste/calidad; GPU elegido por etapa |
| **M3 — Dataset** | Pilot 10k → Alpha 250k vía `.map()` | dataset versionado en Volume + parquet de telemetría |
| **M4 — Conditioning** | E2, E3 | mecanismo y planner elegidos por coste/calidad |
| **M5 — Destilación** | E6, E7 (Klein) | Klein ≥ baseline a paridad de parámetros |
| **M6 — Paper** | redacción, reproducibilidad | dataset + código + informe de costes públicos |

---

## 12. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Las etapas N2–N6 no influyen en N7 (decoración) | E2 mide acoplamiento real; escalar a conditioning fuerte si hace falta |
| Coste de generación masiva se dispara | empezar Smoke/Pilot; elegir GPU por $/unidad; renders intermedios opcionales |
| Cold starts dominan el coste a bajo volumen | `@enter` + snapshots + decisión `keep_warm` basada en E4 |
| La hipótesis de destilación no se sostiene | el dataset y la metodología de coste siguen siendo contribuciones válidas |
| Calidad de la traza del planner barato es baja | E3 compara API vs self-hosted explícitamente |
| Almacenamiento supera 1 TiB con renders intermedios | comprimir WebP, podar intermedios, presupuestar excedente $0.09/GiB/mes |

---

## 13. Decisiones abiertas (a resolver antes de M1)

- **Planner inicial:** ¿API externa (más rápido de validar) o self-hosted (más
  reproducible)? → recomendación: API en Smoke, self-hosted desde Pilot.
- **GPU por defecto del renderer:** H100 como punto de partida; confirmar con E1.
- **Mecanismo de conditioning del MVP:** prompt-enrichment (barato) para M1; subir
  a conditioning espacial en M4.
- **Formato de imagen:** WebP (coste de almacenamiento) vs PNG (sin pérdida para
  métricas). → WebP para finales, PNG para un subconjunto de evaluación.

---

## 14. Lo que de verdad estamos construyendo

La mayoría diría: *"queremos mejores imágenes"*. No es el objetivo.

El objetivo es **construir una representación explícita del pensamiento visual** y
**medir con rigor el coste de producirla y de inferirla**. Si la hipótesis se
sostiene, dejamos de entrenar modelos de imagen y empezamos a entrenar un
**Visual Reasoning Model** — el equivalente visual de los modelos de razonamiento
que aparecieron después de GPT-4 — donde un modelo de 4B–9B exhibe capacidades
que hoy parecen requerir modelos mucho mayores, porque aprende **el proceso que
produce la imagen, no solo la imagen terminada**.
```
