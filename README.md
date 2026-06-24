# vcot — Visual Chain of Thought (V-CoT)

Una representación explícita del **pensamiento visual** y un **modelo de costes de
inferencia** reproducible sobre infraestructura serverless (Modal.com).

La visión completa está en [IDEA.md](IDEA.md). Este repositorio empieza por el
cimiento de menor riesgo del roadmap (M0): la **instrumentación de coste**.

## Estado actual

Implementado:

- [`vcot.telemetry.rates`](src/vcot/telemetry/rates.py) — tabla de tarifas de Modal,
  **única fuente de verdad** (IDEA.md §8.1).
- [`vcot.telemetry.cost_timer`](src/vcot/telemetry/cost_timer.py) — cronómetro que
  mide `compute_s` real y lo convierte a USD (IDEA.md §7.3).
- [`vcot.cli`](src/vcot/cli.py) — estimador de coste por línea de comandos.
- [`modal_app/renderer.py`](modal_app/renderer.py) — **N7 (Final Render)**: genera
  imágenes con FLUX.2 sobre Modal, instrumentado con `cost_timer`. Por defecto usa
  **FLUX.2-klein-9B** (el modelo "Klein" de IDEA.md). Setup y uso en
  [modal_app/README.md](modal_app/README.md).

Pendiente (ver [IDEA.md §10](IDEA.md)): planner N1–N6, `dataset/`, `analysis/`, `train/`.

## Generar imágenes (renderer N7)

Requiere cuenta de Modal y aceptar la licencia gated de FLUX.2 — pasos completos en
[modal_app/README.md](modal_app/README.md). Resumen:

```powershell
pip install -e ".[modal]"
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```

La imagen se descarga a `outputs/` y se imprime la telemetría real de coste.

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
- [src/vcot/cli.py](src/vcot/cli.py)
- [pyproject.toml](pyproject.toml)
