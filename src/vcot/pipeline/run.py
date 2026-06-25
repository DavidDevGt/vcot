"""CLI del planner V-CoT (IDEA.md §2, §6).

Genera la cadena de razonamiento N1–N6 de un prompt usando un LLM **local** y
muestra la traza + telemetría de coste proyectado en Modal::

    python -m vcot.pipeline.run --prompt "a lone astronaut in a gothic cathedral"

Requiere un runtime LLM local OpenAI-compatible escuchando (por defecto Ollama en
http://localhost:11434/v1). Configurable con --base-url/--model o las env vars
VCOT_LLM_BASE_URL / VCOT_LLM_MODEL.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from vcot.pipeline.llm import LocalLLMClient
from vcot.pipeline.planner import DEFAULT_PROJECTED_GPU, Planner
from vcot.pipeline.schemas import STAGE_LABELS


def _summary(trace) -> str:
    lines = ["", "Telemetría por etapa (coste proyectado en " f"{trace.meta.get('projected_gpu')}):"]
    for stage, tele in trace.telemetry.items():
        lines.append(
            f"  {STAGE_LABELS[stage]:<3} {stage:<14} "
            f"{tele.compute_s:6.2f}s  "
            f"{tele.output_tokens:5d} tok  "
            f"{tele.tokens_per_s:6.1f} tok/s  "
            f"${tele.projected_cost_usd:.6f}"
            + (f"  (reintentos: {tele.retries})" if tele.retries else "")
        )
    lines.append(
        f"  {'':<3} {'TOTAL':<14} {trace.total_compute_s:6.2f}s"
        f"{'':>20}${trace.total_projected_cost_usd:.6f}"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-plan",
        description="Genera una traza de razonamiento V-CoT (N1–N6) con un LLM local.",
    )
    parser.add_argument(
        "--prompt",
        default="a lone astronaut inside an abandoned gothic cathedral, "
        "moonlight through stained glass, volumetric fog, cinematic",
        help="Prompt de partida de la cadena.",
    )
    parser.add_argument("--base-url", default=None, help="Endpoint OpenAI-compatible local.")
    parser.add_argument("--model", default=None, help="Modelo local (p.ej. qwen2.5:7b-instruct).")
    parser.add_argument(
        "--gpu",
        default=DEFAULT_PROJECTED_GPU,
        help="GPU de Modal sobre la que proyectar el coste (clave de rates.py).",
    )
    parser.add_argument("--retries", type=int, default=2, help="Reintentos por etapa.")
    parser.add_argument(
        "--out",
        default="outputs",
        help="Carpeta donde guardar la traza JSON (vacío para no guardar).",
    )
    args = parser.parse_args(argv)

    client = LocalLLMClient(base_url=args.base_url, model=args.model)
    planner = Planner(client, projected_gpu=args.gpu, max_retries=args.retries)

    print(f"Planificando con {client.model} @ {client.base_url} …")
    trace = planner.plan(args.prompt)

    print(json.dumps(trace.model_dump(), indent=2, ensure_ascii=False))
    print("\nVisual tokens:\n  " + " ".join(trace.visual_tokens))
    print(_summary(trace))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{trace.id}.trace.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(trace.model_dump(), fh, indent=2, ensure_ascii=False)
        print(f"\nTraza -> {path}")


if __name__ == "__main__":
    main()
