# modal_app — Pipeline V-CoT sobre Modal

**Todo el pipeline corre en Modal serverless GPU**, incluido el LLM del razonamiento
(self-hosted con vLLM, no una API de nube). Dos apps:

- [`planner.py`](planner.py) — **N1–N6 (razonamiento)**: genera la cadena de
  decisiones con un LLM open-weights en GPU. Coste **real** por etapa.
- [`renderer.py`](renderer.py) — **N7 (render)**: genera la imagen con FLUX.2.

Ambas comparten el patrón `@app.cls` + `@modal.enter()` (carga el modelo una vez)
e instrumentan el coste real con el `cost_timer` de `vcot`.

## Renderer N7 (FLUX.2)

Genera imágenes con FLUX.2 en GPU serverless, midiendo coste real por imagen con
el `cost_timer` de `vcot`. Por defecto usa **FLUX.2-klein-9B** (4 pasos, ~29 GB).

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

## Generar imágenes (4 variaciones)

```powershell
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Por defecto genera **4 variaciones** del mismo prompt en un solo batch
(`num_images_per_prompt`): comparten el text-encoding y el contenedor cálido, así
que cuestan mucho menos que 4 llamadas separadas. Se descargan a
`outputs/<id>_0.webp … _3.webp` y se imprime la telemetría (`compute_s`,
`cost_usd` total y `cost_per_image_usd`). Además se persisten en el Volume
`vcot-outputs` (imágenes + línea en `records.jsonl`), el dataset incremental.

Parámetros:

| Flag | Defecto | Qué hace |
|---|---|---|
| `--prompt` | (astronauta) | Texto del render |
| `--variations` | 4 | Nº de variaciones del mismo prompt |
| `--negative` | "" (off) | Prompt negativo; `--negative default` usa el sugerido |
| `--cfg` | 0 = auto (4.0 si hay negativo) | `true_cfg_scale` (CFG real) |
| `--steps` | 0 = el del modelo (4 klein / 28 dev) | Pasos de difusión |
| `--seed` | -1 = aleatorio | Semilla reproducible (batch reproducible) |

> **Prompt negativo en FLUX:** FLUX es *guidance-distilled*; el negativo solo aplica
> con **CFG real** (`true_cfg_scale > 1`, ~2× cómputo). **Verificado (2026-06-25):**
> el `Flux2KleinPipeline` **no expone `negative_prompt`** en la versión actual de
> diffusers → el código lo **detecta por introspección, avisa y lo ignora** (no
> crashea; coste sin cambios, `meta.negative_prompt=""`). Para usar negativo de
> verdad, probá `dev` (`VCOT_MODEL=dev`, no distilled). Ejemplo:
> `modal run modal_app/renderer.py --prompt "..." --negative default`

## Cambiar de modelo / GPU

Por variable de entorno (se leen al construir la app):

```powershell
$env:VCOT_MODEL = "dev"          # FLUX.2-dev 32B (necesita B200 + offload)
$env:VCOT_GPU   = "H100"         # forzar otra GPU (debe existir en rates.py)
modal run modal_app/renderer.py --prompt "..."
```

> `VCOT_GPU` debe coincidir con una clave de
> [`vcot.telemetry.rates`](../src/vcot/telemetry/rates.py) para que el cálculo de
> coste sea correcto (p.ej. `A100-80GB`, `H100`, `B200`, `L40S`).

## Coste orientativo (klein, 4 pasos)

`compute_s × rate`. En A100-80GB (0.000694 $/s) un render de ~2 s ≈ **0.0014 $**.
La cifra exacta la mide `cost_timer` en cada llamada; usa eso, no la estimación.

## Planner N1–N6 (vLLM)

Genera la **cadena de razonamiento** en GPU serverless con un LLM open-weights
(por defecto **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)** en A100-40GB).
Reutiliza la lógica de `vcot.pipeline.Planner` con un cliente vLLM in-process; el
`cost_timer` mide el **coste real** de GPU por etapa.

Qwen3 es híbrido de razonamiento; se ejecuta con **thinking mode desactivado**
(`enable_thinking=False`) — en V-CoT las etapas N1–N6 ya son el razonamiento
explícito, así que no queremos un bloque `<think>` opaco (da JSON limpio y menos
tokens). Requiere **vLLM ≥ 0.8.5** (ya fijado en la imagen).

```powershell
modal run modal_app/planner.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Salida: la traza N1–N6 se descarga a `outputs/<id>.trace.json`, se persiste en el
Volume `vcot-outputs` (`<id>.trace.json` + línea en `traces.jsonl`, el dataset
incremental de *pensamiento visual*) y se imprime la telemetría de coste real por
etapa + los Visual Tokens.

Cambiar de modelo / GPU:

```powershell
$env:VCOT_PLANNER_MODEL = "qwen3-14b"     # necesita A100-80GB
$env:VCOT_PLANNER_GPU   = "A100-80GB"     # debe existir en rates.py
modal run modal_app/planner.py --prompt "..."
```

> Qwen3 no es gated, pero se reutiliza el Secret `huggingface-secret` para evitar
> rate-limits de descarga. El modelo se cachea en el Volume `vcot-hf-cache`.
