"""Reensambla el dataset completo desde los artefactos del Volume (IDEA.md §4.2).

El dataset canónico (traza completa N1→N7 con imágenes ligadas) **siempre se puede
reconstruir** desde lo que persiste en el Volume ``vcot-outputs``:

- ``{id}.trace.json`` — el razonamiento N1–N6 (lo escribe el planner),
- ``records.jsonl`` — el render de cada muestra: ``images`` (con ``sha256``/seed),
  telemetría y el prompt enriquecido (lo escribe el renderer),

unidos por ``id`` (que es el ``trace.id``, porque el render usa
``sample_id = trace.id``). Esto recupera el dataset aunque el ``dataset.jsonl``
local se haya perdido/pisado, y sirve como reconstrucción reproducible a escala
(las imágenes viven en el Volume, no hace falta moverlas para reconstruir el JSONL).

Puro (stdlib), testeable sin Modal.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

from vcot.dataset.seed_prompts import prompt_stratum


def _render_telemetry(record: Dict) -> Optional[Dict]:
    """Adapta la telemetría del renderer a StageTelemetry (como run_pipeline)."""
    tele = record.get("telemetry", {}).get("render", {})
    if not tele:
        return None
    return {
        "compute_s": tele.get("compute_s", 0.0),
        "rate_usd_per_s": tele.get("rate_usd_per_s", 0.0),
        "projected_cost_usd": tele.get("cost_usd", tele.get("projected_cost_usd", 0.0)),
        "projected_gpu": record.get("meta", {}).get("gpu", "unknown"),
    }


def assemble_record(trace: Dict, render: Dict, *, license: str, code_version: Optional[str]) -> Dict:
    """Une razonamiento (trace) + render → traza completa con bloque ``dataset``."""
    rec = dict(trace)
    rec["enriched_prompt"] = render.get("prompt")  # el prompt del render ES el enriquecido
    rec["images"] = render.get("images", [])
    rec["final_image"] = render.get("final_image")
    rec["final_images"] = render.get("final_images", [])
    rec["render"] = _render_telemetry(render)
    rec.setdefault("meta", {})["seed"] = render.get("meta", {}).get("seed")
    rec["dataset"] = {
        "license": license,
        "code_version": code_version,
        "stratum": prompt_stratum(trace.get("prompt", "")),
        "split": None,
        "safety": {},
        "quality": {},
    }
    return rec


def assemble_dataset(
    records_path: str,
    traces_dir: str,
    out_path: str,
    *,
    license: str = "FLUX.2 (non-commercial, BFL) + Qwen3 (Apache-2.0)",
    code_version: Optional[str] = None,
) -> int:
    """Reconstruye ``out_path`` (dataset.jsonl) uniendo records + ``{id}.trace.json``.

    Devuelve el nº de muestras reensambladas. Ignora records sin su ``.trace.json``.
    """
    n = 0
    with open(records_path, encoding="utf-8") as rfh, open(out_path, "w", encoding="utf-8") as ofh:
        for line in rfh:
            line = line.strip()
            if not line:
                continue
            render = json.loads(line)
            trace_path = os.path.join(traces_dir, f"{render['id']}.trace.json")
            if not os.path.exists(trace_path):
                continue
            with open(trace_path, encoding="utf-8") as tfh:
                trace = json.load(tfh)
            rec = assemble_record(trace, render, license=license, code_version=code_version)
            ofh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-assemble",
        description="Reensambla dataset.jsonl desde records.jsonl + {id}.trace.json del Volume.",
    )
    parser.add_argument("--records", default="outputs/records.jsonl")
    parser.add_argument("--traces-dir", default="outputs", help="Carpeta con los {id}.trace.json.")
    parser.add_argument("--out", default="outputs/dataset.jsonl")
    parser.add_argument("--code-version", default=None)
    args = parser.parse_args(argv)

    n = assemble_dataset(
        args.records, args.traces_dir, args.out, code_version=args.code_version
    )
    print(f"{n} muestras reensambladas -> {args.out}")
    print(f"Evaluar:  modal run modal_app/eval.py::evaluate --traces {args.out}")


if __name__ == "__main__":
    main()
