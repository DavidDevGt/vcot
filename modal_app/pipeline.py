"""Pipeline V-CoT completo N1→N7 sobre Modal (IDEA.md §2, §3).

Encadena el **razonamiento** (planner, vLLM) con el **render** (FLUX) pasando por
el enriquecimiento de prompt (§3.1). Ambas etapas corren en sus propias GPUs de
Modal; este app solo orquesta (sin GPU).

Como compone dos apps distintas (imágenes/GPU distintas), las referencia por
nombre, así que **primero hay que desplegarlas**:

    modal deploy modal_app/planner.py
    modal deploy modal_app/renderer.py
    modal run    modal_app/pipeline.py --prompt "a lone astronaut in a gothic cathedral"

El resultado es la traza completa (razonamiento + imagen + coste e2e N1–N7).
"""

from __future__ import annotations

import json
import os

import modal

from vcot.pipeline import run_pipeline
from vcot.pipeline.schemas import VCoTTrace

app = modal.App("vcot-pipeline")

# Referencias a las apps desplegadas (planner.py y renderer.py).
_Planner = modal.Cls.from_name("vcot-planner", "Planner")
_Renderer = modal.Cls.from_name("vcot-renderer", "Renderer")


class _RemotePlanner:
    """Adapta el Planner remoto al contrato `SupportsPlan` de run_pipeline."""

    def __init__(self) -> None:
        self._cls = _Planner()

    def plan(self, prompt: str) -> VCoTTrace:
        return VCoTTrace.model_validate(self._cls.plan.remote(prompt))


@app.local_entrypoint()
def main(
    prompt: str = (
        "a lone astronaut inside an abandoned gothic cathedral, "
        "moonlight through stained glass, volumetric fog, cinematic"
    ),
    negative: str = "",
    out: str = "outputs",
):
    """Ejecuta N1→N7 y guarda traza + 4 imágenes + coste e2e en local.

    `--negative` activa prompt negativo en N7 (CFG real, ~2× coste de render).
    """
    from vcot.reporting.runlog import track_run

    renderer = _Renderer()
    image_holder: dict = {}

    def render_fn(enriched: str, sample_id: str) -> dict:
        # sample_id = trace.id ⇒ imágenes ligadas a la traza ({trace.id}_i.webp).
        record = renderer.render.remote(
            enriched, negative_prompt=negative, sample_id=sample_id
        )  # 4 variaciones
        image_holder["bytes_list"] = record.pop("_image_bytes_list", [])
        return record

    os.makedirs(out, exist_ok=True)
    with track_run(os.path.join(out, "runs.jsonl"), kind="pipeline") as run:
        trace = run_pipeline(prompt, _RemotePlanner(), render_fn)
        run["model"] = str(trace.meta.get("planner"))
        run["gpu"] = trace.render.projected_gpu if trace.render else None
        run["n_items"] = 1
        run["total_cost_usd"] = trace.e2e_cost_usd

    record = trace.model_dump()
    trace_path = os.path.join(out, f"{trace.id}.full.json")
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    # Acumula al dataset local para el informe final.
    with open(os.path.join(out, "traces.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    for i, image_bytes in enumerate(image_holder.get("bytes_list", [])):
        img_path = os.path.join(out, f"{trace.id}_{i}.webp")
        with open(img_path, "wb") as fh:
            fh.write(image_bytes)
        print(f"Variación {i} -> {img_path}")

    print(f"Traza  -> {trace_path}")
    print(f"\nEnriched prompt:\n  {trace.enriched_prompt}")
    print(
        f"\nCoste E2E (N1–N7): ${trace.e2e_cost_usd:.6f}  "
        f"(razonamiento ${trace.total_projected_cost_usd:.6f} + "
        f"render ${trace.render.projected_cost_usd:.6f})"
    )
