# modal_app — Renderer N7 (FLUX.2 sobre Modal)

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

## Generar una imagen

```powershell
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

Salida: la imagen `.webp` se descarga a `outputs/` y se imprime la telemetría
(`compute_s`, `rate_usd_per_s`, `cost_usd`). Además se persiste en el Volume
`vcot-outputs` (imagen + línea en `records.jsonl`), que es el dataset incremental.

Parámetros:

| Flag | Defecto | Qué hace |
|---|---|---|
| `--prompt` | (astronauta) | Texto del render |
| `--steps` | 0 = el del modelo (4 klein / 28 dev) | Pasos de difusión |
| `--seed` | -1 = aleatorio | Semilla reproducible |

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
