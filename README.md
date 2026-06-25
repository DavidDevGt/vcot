# vcot — Visual Chain of Thought

> Una representación explícita del **pensamiento visual**: en lugar de aprender
> `prompt → imagen`, el sistema captura la **secuencia de decisiones** que un
> experto toma antes de que exista un solo píxel — y la vuelve un dataset
> observable, validado y reproducible.

La hipótesis (desarrollada en [IDEA.md](IDEA.md)): la inteligencia visual no vive
en la imagen final, sino en la cadena de decisiones que la precede. Los modelos de
difusión actuales colapsan esa cadena en una sola operación opaca. **V-CoT** la
hace explícita —el análogo visual del Chain-of-Thought textual— para luego
**destilarla** en un modelo pequeño que aprenda a *pensar* visualmente, no solo a
renderizar.

```text
prompt
  │
  ├─ N1  Semantic Plan      qué hay en la escena (semántica pura)
  ├─ N2  Spatial Layout     scene graph: entidades tipadas + bboxes + relaciones
  ├─ N3  Composition        lente, escala del sujeto, líneas guía
  ├─ N4  Lighting           esquema de luz
  ├─ N5  Materials          superficies
  ├─ N6  Color Script       paleta + temperatura
  └─ N7  Render             imagen (FLUX.2), condicionada por la traza
```

Las etapas **N1–N6 son razonamiento puro** (JSON estructurado y validado por
etapa); **N7** dibuja. Cada decisión queda *observable, serializable y criticable*.

---

## Qué hace

- **Razona una traza V-CoT** (N1–N6) con un LLM open-weights, validando cada etapa
  contra su esquema (pydantic) con reintento automático.
- **Renderiza** (N7) con FLUX.2 (4 variaciones por batch), ligando cada imagen a su
  traza por `id` + `sha256`.
- **Genera el dataset a escala** (fan-out N1→N7 sobre Modal serverless GPU) con
  **semilla determinista por muestra** → reproducible bit-a-bit.
- **Evalúa el dataset** (research-grade): alineación prompt↔imagen, preferencia
  humana, estética, **layout-faithfulness** (¿la imagen respeta el scene graph?),
  seguridad y dedup perceptual — con distribuciones y breakdown por estrato.
- **Empaqueta** a shards WebDataset + `manifest.json` + `DATACARD.md`
  (Datasheet for Datasets), y **destila** la traza a un modelo pequeño (SFT).

El backend es agnóstico: el razonamiento corre sobre cualquier endpoint
OpenAI-compatible, así que también se itera **en local** (Ollama / LM Studio / vLLM)
sin tocar la nube.

---

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"          # núcleo + pytest
# pip install -e ".[dev,modal]"  # + cliente Modal para correr planner/renderer/dataset/eval
```

El paquete se instala editable: los imports son `import vcot` (sin prefijo `src.`).

## Quickstart

**Razonar una traza en local** (sin nube; requiere un LLM OpenAI-compatible
escuchando, p.ej. Ollama con `qwen3:8b`):

```powershell
python -m vcot.pipeline.run --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Imprime la cadena N1→N6, los Visual Tokens y guarda la traza en `outputs/<id>.trace.json`.

**En código:**

```python
from vcot.pipeline import Planner, LocalLLMClient

trace = Planner(LocalLLMClient()).plan("a lone astronaut in a gothic cathedral")
print(trace.visual_tokens)
print(trace.layout.relations)   # scene graph: relaciones entre entidades
```

**Pipeline completo (N1→N7) sobre Modal** — setup y comandos en
[modal_app/README.md](modal_app/README.md) y [RUNBOOK.md](RUNBOOK.md):

```powershell
modal deploy modal_app/planner.py
modal deploy modal_app/renderer.py
modal run    modal_app/dataset.py::generate_full --limit 100   # dataset N1→N7 con imágenes
modal run    modal_app/eval.py::evaluate                       # evalúa → dataset.eval.jsonl + quality.json
python -m vcot.dataset.pack outputs/dataset.eval.jsonl --out dataset/   # shards + datacard
```

---

## El artefacto: una traza V-CoT

Cada muestra es un único registro `VCoTTrace` (validado por pydantic). Las etapas
N1–N6 son razonamiento; N7 añade las imágenes ligadas; el bloque `dataset` añade
curación (licencia, split, métricas de calidad, seguridad).

```jsonc
{
  "id": "3eaabdca…",
  "prompt": "an off-duty surgeon in a hospital stairwell at night",
  "semantic_plan": { "subject": "…", "environment": "…", "mood": "…" },
  "layout": {                          // scene graph
    "entities":  [ { "id": "surgeon", "kind": "character", "bbox": [.., .., .., ..] } ],
    "relations": [ { "subject": "surgeon", "predicate": "on", "object": "staircase" } ]
  },
  "composition": { "lens": "35mm", "subject_scale": 0.4, "leading_lines": true },
  "lighting": { "…": "…" }, "materials": { "…": "…" }, "color_script": { "…": "…" },
  "visual_tokens": ["PLAN_SUBJ:surgeon", "…", "RENDER"],
  "images": [ { "path": "…_0.webp", "sha256": "f522…", "idx": 0, "seed": 1679701046 } ],
  "dataset": { "license": "…", "stratum": "portrait", "split": "train",
               "quality": { "faithfulness": 0.5, "detection_coverage": 0.83 },
               "safety": { "nsfw_label": "ok" } }
}
```

> **Hallazgo:** estructurar el razonamiento aumentó la **diversidad** de las
> decisiones más que optimizar el prompt — `subject_scale` pasó de un único valor a
> tres, y aparecieron ~4.2 relaciones espaciales por traza. El sistema empieza a
> proponer composiciones que no se le pidieron de forma explícita (detalle en
> [IDEA.md](IDEA.md)).

---

## Arquitectura

Todo el pipeline pesado corre sobre **Modal serverless GPU**, con modelos
**open-weights** self-hosted (sin APIs de nube): **Qwen3-8B** razona (vLLM),
**FLUX.2-klein-9B** dibuja. La orquestación es agnóstica del backend y testeable
con dobles.

| Módulo | Rol |
|---|---|
| [`vcot.pipeline`](src/vcot/pipeline/) | Razonamiento N1–N6: esquemas validados, `Planner`, Visual Tokens, `enrich`, orquestación N1→N7 |
| [`vcot.dataset`](src/vcot/dataset/) | Prompts semilla + generador estratificado, semilla determinista, conversión a SFT, **pack** (WebDataset + datacard), **assemble** (reconstrucción) |
| [`vcot.eval`](src/vcot/eval/) | **faithfulness** (IoU vs scene graph), splits sin fuga, dedup pHash, gate de calidad, distribuciones por estrato, calibración humana, taxonomía de seguridad |
| [`vcot.train`](src/vcot/train/) | Destilación SFT del modelo pequeño |
| [`vcot.reporting`](src/vcot/reporting/) | Informe profesional (Markdown/JSON/CSV) + ledger de ejecuciones |
| [`vcot.telemetry`](src/vcot/telemetry/) | Instrumentación de cómputo por etapa (latencia, tokens) |
| [`modal_app/`](modal_app/) | Apps Modal: `planner` (N1–N6), `renderer` (N7), `dataset` (fan-out N1→N7), `eval` (scoring GPU) |

**Reproducibilidad:** semilla por muestra derivada del prompt (`derive_seed`),
splits por hash sin fuga, provenance de modelos y `git sha` en el manifiesto. El
dataset se reconstruye desde los artefactos del Volume con
[`vcot.dataset.assemble`](src/vcot/dataset/assemble.py).

---

## Tests

```powershell
pytest
```

El núcleo (esquemas, planner, eval, pack, assemble, dataset) está cubierto con
pruebas puras —sin GPU ni red—; las apps de Modal se validan en una corrida real.

## Documentación

- [IDEA.md](IDEA.md) — la tesis completa, el diseño de las etapas y el plan experimental.
- [RUNBOOK.md](RUNBOOK.md) — guía operativa de punta a punta sobre Modal.
- [modal_app/README.md](modal_app/README.md) — setup de Modal, FLUX.2 (gated) y modelos.

## Licencia

[MIT](LICENSE). Las imágenes generadas con FLUX.2 están sujetas a su licencia
*non-commercial* — respetala al redistribuir el dataset.
