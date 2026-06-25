# modal_app — Pipeline V-CoT sobre Modal

Todo el pipeline pesado corre en **Modal serverless GPU**, con modelos
**open-weights self-hosted** (no APIs de nube). Cuatro apps, cada una desplegable
y ejecutable de forma independiente:

| App | Etapa | Qué hace | GPU por defecto |
|---|---|---|---|
| [`planner.py`](planner.py) | N1–N6 | Razonamiento: genera la cadena de decisiones con un LLM (vLLM) | A100-40GB |
| [`renderer.py`](renderer.py) | N7 | Render con FLUX.2 (4 variaciones/batch), imágenes ligadas por `sha256` | A100-80GB |
| [`dataset.py`](dataset.py) | N1→N7 | Fan-out: genera el dataset completo con imágenes (semilla determinista) | — (orquesta) |
| [`eval.py`](eval.py) | eval | Scoring research-grade del dataset (CLIP, faithfulness, NSFW, dedup…) | L4 |

Las cuatro comparten el patrón `@app.cls` + `@modal.enter()` (carga el modelo una
vez por contenedor) y están **instrumentadas por etapa** (tiempo de cómputo,
tokens). Guía operativa de punta a punta en [../RUNBOOK.md](../RUNBOOK.md).

---

## Setup (una sola vez)

### 1. Modal

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[modal]"     # instala el cliente modal
modal setup                    # abre el navegador para autenticar
```

### 2. Aceptar la licencia de FLUX.2 (modelos gated)

FLUX.2 es **non-commercial y gated**. Inicia sesión en HuggingFace y acepta la
licencia del modelo que vayas a usar:

- Klein: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- Dev:   https://huggingface.co/black-forest-labs/FLUX.2-dev

### 3. Token de HuggingFace → Secret de Modal

Crea un token (rol *read*) en https://huggingface.co/settings/tokens y regístralo
como Secret de Modal con la clave `HF_TOKEN`:

```powershell
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

> Qwen3 no es gated (corre sin token), pero se reutiliza el Secret para evitar
> rate-limits de descarga. Los pesos se cachean en el Volume `vcot-hf-cache`.

> **Consola Windows:** el CLI de Modal imprime UTF-8 y la consola es cp1252 → puede
> crashear. Antes de cualquier `modal run`: `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"`.

---

## Planner — N1–N6 (razonamiento)

Genera la **cadena de razonamiento** en GPU serverless con un LLM open-weights
(por defecto **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**). Reutiliza tal
cual la lógica de `vcot.pipeline.Planner` con un cliente vLLM in-process.

Qwen3 es híbrido de razonamiento; se ejecuta con **thinking mode desactivado**
(`enable_thinking=False`) — en V-CoT las etapas N1–N6 ya son el razonamiento
explícito, así que no queremos un bloque `<think>` opaco (da JSON limpio y menos
tokens). Requiere **vLLM ≥ 0.8.5** (ya fijado en la imagen).

```powershell
modal run modal_app/planner.py::main --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

La traza N1–N6 se guarda en `outputs/<id>.trace.json`, se persiste en el Volume
`vcot-outputs` (`traces.jsonl`, el dataset incremental de *razonamiento*) y se
imprimen los Visual Tokens + la instrumentación por etapa.

`::main` es obligatorio: el planner tiene dos entrypoints (`main` y `generate`).
El segundo genera **solo razonamiento** en paralelo (`.map`) sobre los prompts
semilla — útil para iterar el LLM sin renderizar.

---

## Renderer — N7 (FLUX.2)

Genera imágenes en GPU serverless. Por defecto **FLUX.2-klein-9B** (4 pasos).
Produce **4 variaciones** del mismo prompt en un solo batch
(`num_images_per_prompt`): comparten el text-encoding, mucho más eficiente que 4
llamadas. Cada variación se guarda con su `sha256` y se persiste en el Volume.

```powershell
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

| Flag | Defecto | Qué hace |
|---|---|---|
| `--prompt` | (astronauta) | Texto del render |
| `--variations` | 4 | Nº de variaciones del mismo prompt |
| `--seed` | -1 = aleatorio | Semilla reproducible (batch reproducible) |
| `--steps` | 0 = el del modelo (4 klein / 28 dev) | Pasos de difusión |
| `--negative` | "" (off) | Prompt negativo; `--negative default` usa el sugerido |
| `--cfg` | 0 = auto (4.0 si hay negativo) | `true_cfg_scale` (CFG real) |

> **Prompt negativo en FLUX:** FLUX es *guidance-distilled*; el negativo solo aplica
> con **CFG real** (`true_cfg_scale > 1`). **Verificado:** el `Flux2KleinPipeline`
> **no expone `negative_prompt`** en la versión actual de diffusers → el código lo
> detecta por introspección, avisa y lo ignora (no crashea). Para usar negativo de
> verdad, usá `dev` (`VCOT_MODEL=dev`, no distilled).

---

## Dataset — N1→N7 a escala

Encadena planner → render por cada prompt (fan-out `.map`) y escribe el dataset
**completo con imágenes** en `outputs/dataset.jsonl`. Cada muestra liga su traza a
las 4 imágenes (`sha256` por variación) y lleva una **semilla determinista**
derivada del prompt (`derive_seed` → reproducible bit-a-bit). Requiere desplegar
antes planner y renderer:

```powershell
modal deploy modal_app/planner.py
modal deploy modal_app/renderer.py
modal run    modal_app/dataset.py::generate_full --limit 100
```

Las imágenes quedan en el Volume `vcot-outputs` (no se devuelven por la red →
escala). Para traerlas a local: `modal volume get vcot-outputs / outputs` (no pisa
`dataset.jsonl`). Si se pierde el JSONL, se reconstruye desde el Volume con
`python -m vcot.dataset.assemble`.

---

## Eval — scoring research-grade

Recorre el dataset y puntúa cada muestra con modelos open-weights, apoyándose en el
núcleo puro de `vcot.eval`: **CLIP** (alineación prompt↔imagen), **ImageReward**
(preferencia humana), **aesthetic**, **layout-faithfulness** (OWLv2 detecta las
entidades → IoU vs el scene graph de N2), **NSFW** y **dedup** perceptual (pHash).

```powershell
modal run modal_app/eval.py::evaluate                  # sobre outputs/dataset.jsonl
```

Escribe `dataset.eval.jsonl` (bloque `dataset`: quality/safety/split) y
`quality.json` (distribuciones + breakdown por estrato + provenance de modelos).
Cada scorer está **guardado**: si un modelo no carga, su score queda `null` sin
tumbar la corrida. Lee las imágenes del Volume montado.

> **Faithfulness = línea base.** El render usa solo prompt-enrichment (el layout no
> condiciona el píxel), así que la métrica reporta el *gap*, junto a
> `detection_coverage` para separar "no detectado" de "mal ubicado".

---

## Modelo / GPU

Configurables por variable de entorno (se leen al construir la app):

```powershell
$env:VCOT_PLANNER_MODEL = "qwen3-14b"   # planner (necesita A100-80GB)
$env:VCOT_MODEL         = "dev"          # renderer FLUX.2-dev 32B (B200 + offload)
$env:VCOT_EVAL_GPU      = "A10"          # GPU del eval
```

> El nombre de GPU debe coincidir con una clave de
> [`vcot.telemetry.rates`](../src/vcot/telemetry/rates.py) para que la
> instrumentación por etapa sea correcta (p.ej. `A100-80GB`, `H100`, `L4`).
