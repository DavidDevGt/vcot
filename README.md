# vcot — Visual Chain of Thought (V-CoT)

Una representación explícita del **pensamiento visual** y un **modelo de costes de
inferencia** reproducible sobre infraestructura serverless (Modal.com).

La visión completa está en [IDEA.md](IDEA.md). El pipeline completo N1→N7 corre
sobre **Modal serverless GPU**, con coste instrumentado en cada etapa.

## Estado actual

El pipeline V-CoT está completo de punta a punta:

- [`vcot.telemetry`](src/vcot/telemetry/) — **instrumentación de coste** (M0):
  `rates` (tabla de tarifas de Modal, única fuente de verdad, §8.1) y `cost_timer`
  (mide `compute_s`→USD, §7.3).
- [`vcot.pipeline`](src/vcot/pipeline/) — **razonamiento N1–N6 + cierre del bucle**:
  esquemas validados por etapa, `Planner` (cadena de decisiones con LLM
  open-weights), Visual Tokens, `enrich_prompt` (traza→prompt de render, §3.1) y
  `run_pipeline` (orquestación N1→N7). Corre en Modal ([planner.py](modal_app/planner.py),
  vLLM, coste real); la lógica es agnóstica del backend (cualquier endpoint
  OpenAI-compatible) y también corre local para iterar offline.
- [`modal_app/renderer.py`](modal_app/renderer.py) — **N7 (render)** con FLUX.2
  (default FLUX.2-klein-9B). [`modal_app/pipeline.py`](modal_app/pipeline.py) encadena
  planner→render (N1→N7).
- [`vcot.dataset`](src/vcot/dataset/) — prompts semilla + conversión traza→ejemplos
  SFT; fan-out masivo en [planner.py::generate](modal_app/planner.py) (`.map`, §4).
- [`vcot.analysis`](src/vcot/analysis/) — resumen rápido de coste/latencia por
  etapa con percentiles y proyección a 1k/1M muestras (§8–§9), stdlib puro.
- [`vcot.reporting`](src/vcot/reporting/) — **informe final profesional** (bundle
  Markdown + JSON + CSV con fecha) y **ledger de ejecuciones** (`runs.jsonl`): cada
  proceso queda registrado (modelo, GPU, ítems, coste, duración, estado).
- [`vcot.train`](src/vcot/train/) — prep del dataset de destilación + entrenamiento
  SFT de Klein (E6/M5).

### Verificado en Modal — dataset real de 15 trazas (2026-06-25)

Medido con `cost_timer` sobre 15 trazas (detalle en [IDEA.md §8.3](IDEA.md)):

| Etapa | Modelo / GPU | compute (media) | $/muestra |
|---|---|--:|--:|
| Razonamiento N1–N6 | Qwen3-8B / A100-40GB | 24.1 s | **$0.0141** (rango 0.0074–0.0268) |
| Render N7 (4 variaciones) | FLUX.2-klein-9B / A100-80GB | 9.6 s | $0.0067 ($0.0017/img) |
| **E2E (razonamiento + 4 imgs)** | — | — | **≈ $0.021** |

> **Hallazgo:** el razonamiento cuesta **~2× el render** y **N2 (layout) es ~50%**
> del coste — en V-CoT el cuello es *pensar*, no *dibujar*.

**Calidad (v1 → v2, §8.6):** el scene-graph + prompts anti-plantilla dispararon la
diversidad — `subject_scale` pasó de **1 valor (todo 0.6)** a **3**, las trazas en
español de **2/9 a 0/6**, y aparecieron **4.2 relaciones/traza**. Pendiente:
`leading_lines` sigue 15/15 true y ~27% de bboxes genéricas.

**Falta (requiere más corridas en tu Modal):** dataset a escala, experimentos
E1–E7, entrenar Klein, y conditioning espacial (E2) para que el scene-graph influya
en el píxel (hoy solo prompt-enrichment).

## Razonamiento: generar una traza V-CoT (planner N1–N6)

**En Modal (camino principal)** — el LLM se self-hostea con vLLM en GPU serverless;
el coste medido es real. Setup en [modal_app/README.md](modal_app/README.md):

```powershell
modal run modal_app/planner.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Imprime la cadena N1→N6, los Visual Tokens y la telemetría de coste por etapa;
guarda la traza en `outputs/<id>.trace.json` y en el Volume `vcot-outputs`.

**En local (opcional, para iterar offline sin gastar en Modal)** — apunta a
cualquier LLM OpenAI-compatible (Ollama/vLLM/LM Studio/llama.cpp). Aquí el coste es
*proyectado* (no se factura GPU):

```powershell
ollama serve & ollama pull qwen3:8b
python -m vcot.pipeline.run --prompt "..."                 # default Ollama localhost:11434
$env:VCOT_LLM_BASE_URL = "http://localhost:1234/v1"        # o LM Studio, etc.
```

En código (mismo `Planner`, cualquier cliente):

```python
from vcot.pipeline import Planner, LocalLLMClient

trace = Planner(LocalLLMClient(), projected_gpu="A100-40GB").plan("a lone astronaut")
print(trace.visual_tokens, trace.total_projected_cost_usd)
```

## Generar imágenes (renderer N7)

Requiere cuenta de Modal y aceptar la licencia gated de FLUX.2 — pasos completos en
[modal_app/README.md](modal_app/README.md). Resumen:

```powershell
pip install -e ".[modal]"
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Genera **4 variaciones** del mismo prompt por defecto (un solo batch → comparten
text-encoding, mucho más barato que 4 llamadas). Se descargan a `outputs/<id>_0..3.webp`
y se imprime la telemetría (coste total + `cost_per_image_usd`). Cambiá la cantidad
con `--variations N`.

**Prompt negativo** (opcional): `--negative "blurry, text, ..."` o `--negative default`.
Ojo: FLUX es *guidance-distilled* y **se verificó que `klein` no lo soporta** (el
pipeline no expone `negative_prompt` → se ignora con un warning, sin coste extra).
Para que aplique de verdad hay que usar `dev` (`VCOT_MODEL=dev`), que usa CFG real
(~2× coste de render).

> **¿Por qué `enriched_prompt`, `final_image`, `final_images` y `render` salen `null`
> en una traza del planner?** Porque `planner.py::main`/`generate` solo corre el
> razonamiento N1–N6 (sin render). Esos campos los rellena **N7**, que solo se
> ejecuta vía `pipeline.py` (N1→N7). El esquema es uniforme, por eso aparecen como
> `null` en vez de faltar. (`meta.base_url` es `null` porque el vLLM corre
> in-process, sin endpoint HTTP.)

## Pipeline completo (N1→N7) y flujo de investigación

Todo sobre Modal. Una vez configurado (`modal setup` + Secret de HF):

```powershell
# 1. Pipeline completo de un prompt: razonamiento → render, con coste e2e
modal deploy modal_app/planner.py
modal deploy modal_app/renderer.py
modal run    modal_app/pipeline.py --prompt "a lone astronaut in a gothic cathedral"

# 2. Generar el dataset en paralelo (fan-out con .map sobre los prompts semilla)
#    Acumula outputs/traces.jsonl + registra la corrida en outputs/runs.jsonl
modal run modal_app/planner.py::generate --limit 20

# 3. Informe final profesional (Markdown + JSON + CSV con fecha) — IDEA.md §8/§9
python -m vcot.reporting outputs/ --out reports/
#    Vista rápida sin bundle:
python -m vcot.analysis.aggregate outputs/traces.jsonl

# 4. Preparar el dataset de destilación de Klein (local) y entrenar (GPU)
python -m vcot.train.distill outputs/traces.jsonl --out outputs/sft.jsonl
python -m vcot.train.distill outputs/traces.jsonl --out outputs/sft.jsonl --train   # requiere extra `train`
```

> Cada proceso (planner, render, dataset, pipeline) deja constancia en
> `outputs/runs.jsonl` (quién, modelo/GPU, ítems, coste, duración, estado). El
> informe de `vcot.reporting` lo incluye como sección de auditoría y escribe un
> bundle con fecha en `reports/<timestamp>/` (+ `reports/latest.md`).

## Instalación (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"        # núcleo + pytest
# pip install -e ".[dev,ml]"   # añade torch/transformers cuando exista el pipeline
```

El paquete se instala en modo editable, así que los imports son `import vcot` (sin
prefijos `src.`).

## Uso

Estimador de coste de render:

```powershell
python -m vcot.cli --gpu H100 --compute-s 6 --samples 1000
# o, tras instalar, el script de consola:
vcot --gpu A100-80GB --compute-s 10 --samples 1000
```

Cronómetro de coste en código (IDEA.md §7.3):

```python
from vcot.telemetry import cost_timer

with cost_timer(gpu="H100") as t:
    image = run_flux(...)
telemetry["render"] = t.as_dict()   # {"compute_s", "rate_usd_per_s", "cost_usd"}
```

## Tests

```powershell
pytest
```

## Archivos clave

- [src/vcot/telemetry/rates.py](src/vcot/telemetry/rates.py)
- [src/vcot/telemetry/cost_timer.py](src/vcot/telemetry/cost_timer.py)
- [src/vcot/pipeline/schemas.py](src/vcot/pipeline/schemas.py) — etapas N1–N6
- [src/vcot/pipeline/planner.py](src/vcot/pipeline/planner.py) — cadena V-CoT
- [src/vcot/cli.py](src/vcot/cli.py)
- [pyproject.toml](pyproject.toml)
