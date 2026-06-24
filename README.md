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

Pendiente (ver [IDEA.md §10](IDEA.md)): `modal_app/`, `dataset/`, `analysis/`, `train/`.

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
