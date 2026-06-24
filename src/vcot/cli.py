"""CLI de vcot.

Por ahora expone un estimador de coste de render que usa la tabla de tarifas
(única fuente de verdad). Sirve como sanity-check de la instrumentación de §8
sin necesidad de tocar Modal::

    python -m vcot.cli --gpu H100 --compute-s 6 --samples 1000
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from vcot.telemetry.rates import gpu_rate


def estimate_cost(gpu: str, compute_s: float, samples: int = 1) -> float:
    """Coste estimado en USD de ``samples`` renders de ``compute_s`` cada uno."""
    return gpu_rate(gpu) * compute_s * samples


def _format(gpu: str, compute_s: float, samples: int) -> str:
    per_sample = estimate_cost(gpu, compute_s, 1)
    total = per_sample * samples
    return (
        f"GPU {gpu}: {gpu_rate(gpu):.6f} $/s\n"
        f"  {compute_s:g} s/muestra  ->  {per_sample:.6f} $/muestra\n"
        f"  {samples} muestras       ->  {total:.2f} $"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot",
        description="Estimador de coste de inferencia sobre Modal (IDEA.md §8).",
    )
    parser.add_argument("--gpu", default="H100", help="GPU de Modal (p.ej. H100, A100-80GB)")
    parser.add_argument(
        "--compute-s", type=float, default=6.0, help="Segundos de cómputo por muestra"
    )
    parser.add_argument(
        "--samples", type=int, default=1, help="Número de muestras a estimar"
    )
    args = parser.parse_args(argv)
    print(_format(args.gpu, args.compute_s, args.samples))


if __name__ == "__main__":
    main()
