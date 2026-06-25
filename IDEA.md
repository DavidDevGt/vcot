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
  1. Usar Qwen3-8B como *planner* + FLUX.2 como renderer para generar un
     **dataset sintético de razonamiento visual** a gran escala.
  2. **Destilar** ese proceso en un modelo pequeño ("Klein", 4B–9B) que aprende a
     *pensar* visualmente, no solo a renderizar.
  3. **Medirlo todo**: latencia por etapa, throughput, memoria GPU, tokens/s,
     calidad — y sobre todo **coste por muestra y por capacidad** en Modal.
- **Infra:** Modal.com (serverless GPU). Cada etapa del pipeline es una `Function`
  independiente, lo que permite atribuir coste **por etapa** y elegir el GPU
  óptimo para cada una.
- **Medido (Modal, 2026-06-25 · 15 trazas):** razonamiento N1–N6 media
  **$0.0141/traza** + **4 imágenes** $0.0067 ≈ **$0.021/muestra**. El razonamiento
  cuesta ~2× el render y **N2 (layout) es ~50%** del coste — el coste vive en la
  cadena de decisiones (§8.3). El salto v1→v2 disparó la **diversidad**
  (`subject_scale` 1→3 valores; español 2/9→0/6; relaciones 0→4.2/traza) — §8.6.
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

**N2 — Spatial Layout** (scene graph: entidades **tipadas** + **relaciones**):

```json
{
  "canvas": [1024, 1024],
  "entities": [
    {"id": "astronaut", "kind": "character",  "bbox": [0.35, 0.45, 0.65, 0.95], "z": 1},
    {"id": "cathedral", "kind": "background", "bbox": [0.00, 0.00, 1.00, 1.00], "z": 4},
    {"id": "moonlight", "kind": "light",      "bbox": [0.10, 0.00, 0.45, 0.40], "z": 3},
    {"id": "fog",       "kind": "atmosphere", "bbox": [0.00, 0.70, 1.00, 1.00], "z": 0}
  ],
  "relations": [
    {"subject": "astronaut", "predicate": "inside",         "object": "cathedral"},
    {"subject": "moonlight", "predicate": "passes_through", "object": "cathedral"},
    {"subject": "astronaut", "predicate": "illuminated_by", "object": "moonlight"}
  ]
}
```

> `kind` separa **objeto/personaje** (con bbox real) de **luz/atmósfera/sombra**
> (un efecto no es un objeto de pantalla completa); solo `background` puede ocupar
> todo el lienzo. Las `relations` son el scene graph — el *cómo se relacionan* las
> cosas, no solo dónde están. Esto ataca el *schema filling* (decisiones genéricas)
> empujando hacia *reasoned decomposition*.

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
PLAN_SUBJ:a-lone-astronaut  ENV:a-gothic-cathedral  CAM:low-angle-wide-shot
AT:astronaut:character:center  AT:moonlight:light:top  REL:astronaut:inside:cathedral
REL:moonlight:passes_through:cathedral  LENS_35MM  THIRDS  SCALE:0.6
LIGHT_KEY:moonlight-...  CONTRAST_HIGH  COLOR_COLD  SAT_LOW  PAL:#2e2e2e  RENDER
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

### 4.1 Roles (implementado)

- **Planner (N1–N6):** **Qwen3-8B** open-weights, self-hosted con vLLM en GPU de
  Modal (A100-40GB), con *thinking mode* desactivado (en V-CoT las etapas N1–N6 ya
  son el razonamiento explícito). Coste real medido por etapa (§8.3). La lógica es
  agnóstica del backend (cualquier endpoint OpenAI-compatible), así que también
  corre contra un LLM local (Ollama/LM Studio) para iterar offline sin gastar.
- **Renderer (N7):** **FLUX.2-klein-9B** (4 pasos) en GPU de Modal (A100-80GB).
  Genera **4 variaciones** del mismo prompt por defecto (un solo batch
  `num_images_per_prompt` → comparten text-encoding, mucho más barato que 4
  llamadas), guardadas en `final_images`. **Prompt negativo** opcional (`--negative`):
  FLUX es *guidance-distilled* → solo aplica con CFG real (~2× coste); verificado que
  `klein` no lo soporta (se ignora con warning), usar `dev` para que tenga efecto.
- **Critic (opcional, §5.3):** VLM que evalúa cada etapa y permite auto-corrección
  (pendiente).

### 4.2 Registro por muestra

Cada muestra se persiste como un único registro `VCoTTrace` (lo que escribe el
planner; valores reales de una corrida):

```json
{
  "id": "87de8fda...",
  "prompt": "a lone astronaut in a gothic cathedral, moonlight",
  "semantic_plan": { "subject": "...", "environment": "...", "...": "..." },
  "layout": { "canvas": [1024, 1024], "entities": [ "..." ] },
  "composition": { "...": "..." }, "lighting": { "...": "..." },
  "materials": { "...": "..." }, "color_script": { "...": "..." },
  "visual_tokens": ["PLAN_SUBJ:a-lone-astronaut", "...", "RENDER"],
  "enriched_prompt": null,     // null en traza solo-planner; lo rellena el pipeline (§3.1)
  "final_image": null,         // variación principal tras N7 (vía pipeline.py)
  "final_images": [],          // las 4 variaciones (rutas en el Volume)
  "render": null,              // telemetría de N7 si hubo render
  "telemetry": {
    "layout": {
      "compute_s": 6.59, "rate_usd_per_s": 0.000583,
      "projected_cost_usd": 0.003843, "projected_gpu": "A100-40GB",
      "input_tokens": 412, "output_tokens": 254,
      "tokens_per_s": 38.5, "retries": 0
    }
    // ... una entrada por etapa N1–N6
  },
  "meta": {
    "planner": "Qwen/Qwen3-8B", "projected_gpu": "A100-40GB",
    "created_at": "2026-06-25T01:54:02+00:00",
    "model_load_s": 49.6, "execution": "modal"
  }
}
```

> Los campos de N7 (`enriched_prompt`, `final_image`, `final_images`, `render`) van
> `null` en una traza **solo-planner** (N1–N6, p.ej. `planner.py::generate`); se
> rellenan al correr el pipeline completo N1→N7 (`pipeline.py`). El esquema es
> uniforme, por eso aparecen como `null` en vez de faltar.
>
> En Modal `projected_cost_usd` es el **coste real** (`compute_s × tarifa de la GPU
> usada`); se llama "projected" porque el mismo código corre en local para iterar,
> donde sí es una proyección. El valor del dataset no son las imágenes — es la
> **estructura de decisiones**. `ImageNet = imágenes`. `V-CoT = decisiones visuales.`

### 4.3 Escalas objetivo

| Fase | Muestras | Propósito |
|------|----------|-----------|
| Smoke | 100 | Validar pipeline, instrumentación y coste real medido |
| Pilot | 10 000 | Curvas coste/calidad, elegir GPU por etapa |
| Alpha | 250 000 | Suficiente para destilación inicial de Klein |
| Beta | 1 000 000+ | Dataset de referencia |

### 4.4 Cómo se genera y dónde se guarda (real)

**Generar el dataset** (fan-out paralelo con `.map`, §7.2):

```powershell
# lote pequeño primero (mirá el coste), luego los 36 prompts semilla
modal run modal_app/planner.py::generate --limit 5
modal run modal_app/planner.py::generate
# tus propios prompts (uno por línea):
modal run modal_app/planner.py::generate --prompts-file mis_prompts.txt
```

**Dónde queda todo:**

| Artefacto | Ubicación | Qué es |
|---|---|---|
| Dataset (trazas) | Volume `vcot-outputs` → `traces.jsonl` + `<id>.trace.json` | persistente en Modal, incremental |
| Copia local | `outputs/traces.jsonl` | el `generate` lo acumula también en tu disco |
| Ledger de corridas | `outputs/runs.jsonl` | cada proceso: modelo, GPU, ítems, coste, duración, estado |
| Imágenes (N7) | Volume `vcot-outputs` → `<id>.webp` | del renderer / pipeline |
| Informe final | `reports/<timestamp>/` (+ `reports/latest.md`) | Markdown + JSON + CSV con fecha |

**Traza completa con imagen (N1→N7)** — usar el pipeline (necesita deploy previo):

```powershell
modal deploy modal_app/planner.py ; modal deploy modal_app/renderer.py
modal run modal_app/pipeline.py --prompt "..."     # traza con final_image + coste e2e
```

**Bajar el dataset del Volume y sacar el informe:**

```powershell
modal volume get vcot-outputs traces.jsonl outputs/traces.jsonl
python -m vcot.reporting outputs/ --out reports/
```

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
extremo a extremo. Todo se vuelca a la telemetría de cada muestra (§4.2), al
**ledger de ejecuciones** (`runs.jsonl`) y a un **informe agregado** con fecha
(Markdown/JSON/CSV vía `vcot.reporting`; export a parquet pendiente).

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
# Dos apps (imágenes y GPU distintas), patrón @app.cls + @enter:
app = modal.App("vcot-planner")    #  + modal.App("vcot-renderer")

@app.cls(gpu="A100-40GB", ...)  Planner   # N1–N6 · Qwen3-8B en vLLM (coste real)
@app.cls(gpu="A100-80GB", ...)  Renderer  # N7 · FLUX.2-klein
# dataset:  Planner().plan.map(prompts)   # fan-out paralelo (§4.4)
# (pendiente) Critic (self-correction, §5.3)
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

### 8.3 Mediciones reales (Modal, 2026-06-25 · dataset de 15 trazas)

> Números **medidos** en Modal con el `cost_timer` sobre un dataset real de
> **15 trazas** (9 v1 + 6 v2 scene-graph). Planner = **Qwen3-8B** (vLLM,
> *thinking off*) en A100-40GB. Render = **FLUX.2-klein-9B** (4 pasos, 1024²) en
> A100-80GB. Contenedor cálido (`load_s` amortizado aparte).

**Planner N1–N6 — media por etapa (15 trazas):**

| Etapa | compute_s | tokens out | $/traza |
|---|---:|---:|---:|
| N1 semantic_plan | 5.14 | 108 | 0.00300 |
| N2 layout | 12.17 | 414 | 0.00709 |
| N3 composition | 1.29 | 43 | 0.00075 |
| N4 lighting | 1.21 | 40 | 0.00070 |
| N5 materials | 2.10 | 70 | 0.00122 |
| N6 color_script | 2.22 | 76 | 0.00129 |
| **Total** | **24.1** | **~751** | **0.01406** |

Coste por traza: media **$0.0141**, rango **$0.0074–$0.0268**. **N2 (layout) es el
cuello**: ~50% del compute y del coste. El scene-graph (v2) sube el layout de
$0.0044 (239 tok) a $0.0112 (676 tok) → una traza v2 cuesta ~**60% más** que v1
($0.0182 vs $0.0113). Es el precio de las relaciones — la señal valiosa.

**Render N7 — FLUX.2-klein, A100-80GB, 4 variaciones por batch:**

| variaciones | compute_s | $/batch | $/imagen |
|---:|---:|---:|---:|
| 4 | 9.59 | 0.00665 | **0.00166** |

(1 imagen suelta ≈ 3.2 s / $0.0022; el batch de 4 baja el $/imagen a $0.0017 al
compartir text-encoding y contenedor.)

**Coste por muestra completa (razonamiento + 4 imágenes):**

```text
≈ 0.0141 (N1–N6)  +  0.0067 (4 imgs)  ≈  0.021 $/muestra
```

| Escala | razonamiento | + 4 imágenes |
|---|---:|---:|
| 1 000 | $14 | $21 |
| 10 000 | $141 | $207 |
| 250 000 | $3 515 | $5 180 |
| 1 000 000 | $14 060 | $20 700 |

> **Hallazgo principal:** el **razonamiento (N1–N6) cuesta ~2× el render de 4
> imágenes** ($0.0141 vs $0.0067) y dentro del razonamiento **N2 (layout) es ~50%**.
> En V-CoT el cuello de coste es *pensar*, no *dibujar*. Refuerza la tesis: la
> inteligencia (y el coste) vive en la cadena de decisiones, no en el píxel final.

**Cold start (facturado; se amortiza con contenedores cálidos / `min_containers`):**

| Modelo | model_load_s | $ (1ª vez) |
|---|---:|---:|
| Qwen3-8B (A100-40GB) | 39–76 s | 0.023–0.044 |
| FLUX.2-klein (A100-80GB) | 148.9 s (incl. descarga ~29 GB) | ~0.103 |

> El `load_s` cae a ≈0 por muestra en lotes grandes (un contenedor cálido sirve
> muchas trazas). Cuantificar ese punto de equilibrio es el experimento E4.
> Sigue pendiente E1 (barrido de GPUs) para el óptimo $/calidad: el más caro por
> segundo (B200) puede ser el **más barato por unidad de trabajo** si acelera lo
> suficiente — la métrica es coste por trabajo, no por hora.

> Con **renders intermedios** (conditioning fuerte, §3.3) el coste de render se
> multiplica por ~(1 + nº de pasos intermedios): palanca de coste de primer orden.

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

### 8.6 Diversidad: de *schema filling* a *reasoned decomposition*

Medido sobre las 15 trazas, el salto v1 → v2 (scene-graph + prompts anti-plantilla
+ inglés forzado) es nítido:

| Señal | v1 (n=9) | v2 (n=6) |
|---|---|---|
| `subject_scale` distintos | **1** (todo 0.6) | **3** (0.25 / 0.3 / 0.4) |
| `lens` | 35mm casi siempre (8/9) | 24mm / 35mm variado |
| `rule_of_thirds` = true | 9/9 | 5/6 |
| trazas en **español** (bug) | 2/9 | **0/6** |
| relaciones / traza | 0 | **4.2** |
| entidades / traza | sin tipo | 5.8 (tipadas) |
| bbox genéricas (área>0.6, no-bg) | — | 27% |

v1 era un **vector casi constante** (la composición no aportaba información: *schema
filling*). v2 produce decisiones específicas por escena (*reasoned decomposition*),
con un scene-graph válido (predicados `inside`/`passes_through`/`casts_shadow_on`
dominantes, efectos tipados como `light`/`atmosphere`/`shadow`).

**Templating residual (pendiente):** `leading_lines` sigue en **15/15 true** y ~27%
de bboxes aún son genéricas — el razonamiento geométrico fino y algunos booleanos
todavía caen en defaults. Próximo objetivo de calidad.

> Nota: hoy el layout influye poco en el píxel final porque solo usamos
> *prompt-enrichment* (§3.1). Su valor real (controlabilidad espacial) se mide en
> E2 con conditioning fuerte — por eso "el layout aporta poco a la imagen" es cierto
> bajo el conditioning actual, no una conclusión sobre el scene-graph en sí.

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

## 10. Estructura del repositorio (implementada)

```text
vcot/
  src/vcot/
    telemetry/
      rates.py          # tabla de tarifas (§8.1), única fuente de verdad      [hecho]
      cost_timer.py     # cronómetro + cálculo de coste por etapa (§7.3)       [hecho]
    pipeline/
      schemas.py        # esquemas pydantic de N1–N6 + VCoTTrace (§2.1)        [hecho]
      prompts.py        # prompts por etapa (chain-of-thought)                 [hecho]
      llm.py            # LLMClient (local OpenAI-compatible) + fake           [hecho]
      planner.py        # genera la cadena N1–N6 con coste por etapa           [hecho]
      visual_tokens.py  # traza ↔ tokens (§2.2)                                [hecho]
      enrich.py         # traza → prompt de render (§3.1)                      [hecho]
      pipeline.py       # orquestación N1→N7                                    [hecho]
    dataset/
      seed_prompts.py   # prompts semilla (§4.3)                              [hecho]
      sft.py            # traza → ejemplos de destilación (§5.1)              [hecho]
    analysis/
      aggregate.py      # telemetría JSONL → informe de coste/latencia (§8)   [hecho]
    train/
      distill.py        # prep del dataset SFT + entrenamiento de Klein (§5)  [hecho]
  modal_app/
    planner.py          # N1–N6 sobre Modal (vLLM) + fan-out .map (§4)         [hecho]
    renderer.py         # N7 FLUX.2 sobre Modal                                [hecho]
    pipeline.py         # N1→N7 end-to-end sobre Modal                         [hecho]
  IDEA.md               # este documento
```

> Pendiente sobre este esqueleto: `critic.py` (self-correction, E7), mecanismos de
> conditioning más fuertes que prompt-enrichment (§3.2–3.3), y export a parquet en
> `analysis` (hoy CSV/stdlib). `rates.py` es la **única fuente de verdad** de las
> tarifas: todo informe de coste importa desde ahí.

---

## 11. Roadmap

| Hito | Contenido | Criterio de éxito | Estado |
|---|---|---|---|
| **M0 — Setup** | App Modal, Volume, Secret, `cost_timer`, `rates.py` | una imagen generada con coste real medido | ✅ **hecho** (planner + render corridos en Modal, coste real medido) |
| **M1 — Pipeline V-CoT** | N1–N7 funcionando, registro completo por muestra | 100 muestras (Smoke) con telemetría completa | ✅ etapas N1–N6 y N7 verificadas en GPU; falta correr el lote Smoke completo |
| **M2 — Medición** | E1, E4, E5 | fronteras de Pareto coste/calidad; GPU elegido por etapa | pendiente (necesita GPUs) |
| **M3 — Dataset** | Pilot 10k → Alpha 250k vía `.map()` | dataset versionado en Volume + telemetría | fan-out listo; falta correrlo |
| **M4 — Conditioning** | E2, E3 | mecanismo y planner elegidos por coste/calidad | baseline (enrich) listo; falta E2/E3 |
| **M5 — Destilación** | E6, E7 (Klein) | Klein ≥ baseline a paridad de parámetros | prep listo; falta el entrenamiento |
| **M6 — Paper** | redacción, reproducibilidad | dataset + código + informe de costes públicos | pendiente |

> **Estado global:** pipeline N1→N7 + dataset + análisis + prep de destilación
> implementado y testeado (61 tests). **Verificado en GPUs reales de Modal el
> 2026-06-25**: planner (Qwen3-8B) y render (FLUX.2-klein) corren y reportan coste
> real (§8.3). Lo que falta es **escala y experimentos**: generar el dataset
> completo, correr E1–E7 y entrenar Klein.

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

## 13. Decisiones (resueltas en M0/M1)

- **Planner:** ✅ self-hosted **Qwen3-8B** en vLLM sobre Modal (no API de nube),
  backend-agnóstico para iterar en local. E3 sigue abierto: comparar tamaños/calidad.
- **GPU:** ✅ planner A100-40GB, renderer A100-80GB. El barrido para el óptimo
  $/calidad es E1 (pendiente).
- **Conditioning del MVP:** ✅ prompt-enrichment (`enrich.py`, §3.1). Subir a
  conditioning espacial es E2/M4.
- **Modelo de render:** ✅ FLUX.2-klein-9B (4 pasos) por defecto; FLUX.2-dev (32B)
  vía `VCOT_MODEL=dev`.
- **Formato de imagen:** ✅ WebP (q92) para finales. PNG para un subconjunto de
  evaluación queda pendiente.

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
