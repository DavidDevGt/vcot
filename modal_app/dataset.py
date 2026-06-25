"""Dataset N1→N7 a escala: fan-out razonamiento + render (IDEA.md §4.3, §3).

Esto es lo que ``planner.py::generate`` **no** hace: genera el dataset con
**imágenes**, no solo trazas de razonamiento. Por cada prompt orquesta la cadena
completa N1→N7 (planner → enrich → render) y escribe un record con la traza
**ligada a sus N imágenes** (``images`` con ``sha256`` por variación) + un bloque
``dataset`` de provenance (licencia + git sha).

Compone las dos apps GPU desplegadas (``vcot-planner`` y ``vcot-renderer``) por
nombre, así que **primero hay que desplegarlas**:

    modal deploy modal_app/planner.py
    modal deploy modal_app/renderer.py
    modal run    modal_app/dataset.py::generate_full --limit 100   # Smoke 100

La orquestación corre en un contenedor CPU barato (sin GPU): solo llama a las
funciones GPU remotas. El fan-out (`.map`) paraleliza una muestra completa por
invocación, de modo que el throughput escala con la concurrencia de Modal.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import modal

#: Licencia efectiva de cada muestra (render + planner). FLUX.2 es non-commercial
#: gated; Qwen3 es Apache-2.0. Se registra en el datacard (Fase 4).
SAMPLE_LICENSE = "FLUX.2 (non-commercial, BFL) + Qwen3 (Apache-2.0)"

# Orquestador sin GPU: solo necesita el paquete `vcot` (pipeline + pydantic).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pydantic>=2")
    .add_local_python_source("vcot")
)

app = modal.App("vcot-dataset", image=image)

# Referencias a las apps GPU desplegadas (planner.py y renderer.py).
_Planner = modal.Cls.from_name("vcot-planner", "Planner")
_Renderer = modal.Cls.from_name("vcot-renderer", "Renderer")


class _RemotePlanner:
    """Adapta el Planner remoto al contrato ``SupportsPlan`` de run_pipeline.

    Lleva la ``seed`` determinista de la muestra para que la cadena N1–N6 sea
    reproducible (mismo prompt → misma traza).
    """

    def __init__(self, seed: int | None = None) -> None:
        from vcot.pipeline.schemas import VCoTTrace

        self._cls = _Planner()
        self._VCoTTrace = VCoTTrace
        self._seed = seed

    def plan(self, prompt: str):
        return self._VCoTTrace.model_validate(self._cls.plan.remote(prompt, seed=self._seed))


@app.function(timeout=60 * 60)
def plan_and_render(prompt: str, negative: str = "") -> dict:
    """Cadena completa N1→N7 de **un** prompt; devuelve el record + bytes WebP.

    Usa ``run_pipeline`` (misma lógica que ``modal_app/pipeline.py``) con un
    ``render_fn`` que llama al renderer remoto pasándole ``sample_id = trace.id``
    para que las imágenes queden ligadas a la traza. La **semilla determinista**
    (``derive_seed(prompt)``) se propaga a planner y renderer ⇒ muestra
    reproducible bit-a-bit (queda en ``meta.seed`` y ``images[].seed``).
    """
    from vcot.dataset.seedgen import derive_seed
    from vcot.pipeline import run_pipeline

    seed = derive_seed(prompt)
    renderer = _Renderer()

    def render_fn(enriched: str, sample_id: str) -> dict:
        # El renderer ya persiste las imágenes en el Volume `vcot-outputs`; NO
        # devolvemos bytes por el .map (no escala a 10k+). Las imágenes se traen
        # del Volume con `modal volume get vcot-outputs` cuando se necesiten local.
        return renderer.render.remote(
            enriched, negative_prompt=negative, sample_id=sample_id, seed=seed
        )

    trace = run_pipeline(prompt, _RemotePlanner(seed), render_fn)
    record = trace.model_dump()
    record["meta"]["seed"] = seed  # reproducibilidad explícita de la muestra
    return record


def _e2e_cost(record: dict) -> float:
    """Coste e2e (N1–N7) desde el record serializado (la @property no se dumpea)."""
    reasoning = sum(
        float(t.get("projected_cost_usd", 0.0) or 0.0)
        for t in record.get("telemetry", {}).values()
    )
    render = float((record.get("render") or {}).get("projected_cost_usd", 0.0) or 0.0)
    return reasoning + render


def _git_sha() -> str:
    """git sha corto del código que generó el dataset (provenance)."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 - fuera de un repo git / git ausente
        return "unknown"


@app.local_entrypoint()
def generate_full(
    prompts_file: str = "",
    limit: int = 0,
    negative: str = "",
    out: str = "outputs",
):
    """Genera el dataset **con imágenes** (N1→N7) en paralelo y lo guarda en local.

        modal run modal_app/dataset.py::generate_full --limit 100   # Smoke 100
        modal run modal_app/dataset.py::generate_full --prompts-file mis_prompts.txt

    Sin ``--limit`` usa el núcleo curado; con ``--limit N`` usa la expansión
    estratificada determinista (``vcot.dataset.generate_prompts``). Cada muestra
    queda en ``outputs/traces.jsonl`` (+ imágenes ``{trace.id}_i.webp``); la
    corrida se registra en ``outputs/runs.jsonl``.
    """
    from vcot.dataset import SEED_PROMPTS, generate_prompts, prompt_stratum
    from vcot.reporting.runlog import track_run

    if prompts_file:
        with open(prompts_file, encoding="utf-8") as fh:
            prompts = [ln.strip() for ln in fh if ln.strip()]
        if limit:
            prompts = prompts[:limit]
    elif limit:
        prompts = generate_prompts(limit)
    else:
        prompts = list(SEED_PROMPTS)

    os.makedirs(out, exist_ok=True)
    # Nombre DISTINTO de `traces.jsonl` a propósito: el planner escribe
    # `traces.jsonl` (solo razonamiento) en el Volume, así que `modal volume get`
    # lo pisaría. El dataset completo (con imágenes) vive en `dataset.jsonl`.
    traces_path = os.path.join(out, "dataset.jsonl")
    code_version = _git_sha()
    print(f"Generando {len(prompts)} muestras N1→N7 (con imágenes) en Modal …")

    n = 0
    total_cost = 0.0
    n_images = 0
    with track_run(
        os.path.join(out, "runs.jsonl"),
        kind="dataset_full",
        params={"n_prompts": len(prompts), "code_version": code_version},
    ) as run:
        with open(traces_path, "a", encoding="utf-8") as fh:
            for index, record in enumerate(
                plan_and_render.map(prompts, kwargs={"negative": negative}, return_exceptions=True),
                start=1,
            ):
                if isinstance(record, Exception):
                    prompt = prompts[index - 1] if index <= len(prompts) else "<unknown>"
                    print(
                        f"WARNING: muestra #{index} falló: {prompt!r}\n"
                        f"  error: {type(record).__name__}: {record}",
                        file=sys.stderr,
                    )
                    continue

                # Provenance/curación (split/safety/quality los rellenan fases 3).
                record["dataset"] = {
                    "license": SAMPLE_LICENSE,
                    "code_version": code_version,
                    "stratum": prompt_stratum(record["prompt"]),
                    "split": None,
                    "safety": {},
                    "quality": {},
                }
                n_images += len(record.get("images", []))
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                total_cost += _e2e_cost(record)

        run["n_items"] = n
        run["total_cost_usd"] = total_cost

    avg = total_cost / n if n else 0.0
    print(
        f"\n{n} muestras · {n_images} imágenes · coste e2e ≈ ${total_cost:.4f} "
        f"(${avg:.6f}/muestra) · dataset en {traces_path}"
    )
    print("Imágenes en el Volume vcot-outputs. Para traerlas a local (para pack):")
    print(f"  modal volume get vcot-outputs / {out}   # NO pisa dataset.jsonl")
    print(f"Evaluar:  modal run modal_app/eval.py::evaluate --traces {traces_path}")
