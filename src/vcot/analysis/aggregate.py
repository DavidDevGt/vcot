"""Agrega telemetría de trazas V-CoT a un informe de coste/latencia (IDEA.md §8, §9).

Entrada: un JSONL de registros ``VCoTTrace.model_dump()`` (lo que escriben el
planner y el pipeline en el Volume). Salida: medias, percentiles p50/p90 por etapa
y proyecciones de coste a escala. Sin dependencias externas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Dict, List, Optional, Sequence

from vcot.pipeline.schemas import STAGE_LABELS, STAGE_MODELS


def load_jsonl(path: str) -> List[dict]:
    """Carga un JSONL, ignorando líneas en blanco."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct(values: Sequence[float], p: float) -> float:
    """Percentil ``p`` (0–100) con interpolación lineal. ``[]`` → 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: List[dict]) -> dict:
    """Resumen de coste/latencia por etapa y de la cadena completa."""
    n = len(records)
    per_stage: Dict[str, dict] = {}

    for stage in STAGE_MODELS:
        computes = [r["telemetry"][stage]["compute_s"] for r in records if stage in r.get("telemetry", {})]
        costs = [r["telemetry"][stage]["projected_cost_usd"] for r in records if stage in r.get("telemetry", {})]
        toks = [r["telemetry"][stage].get("output_tokens", 0) for r in records if stage in r.get("telemetry", {})]
        per_stage[stage] = {
            "label": STAGE_LABELS[stage],
            "compute_s_mean": _mean(computes),
            "compute_s_p50": _pct(computes, 50),
            "compute_s_p90": _pct(computes, 90),
            "output_tokens_mean": _mean(toks),
            "cost_mean": _mean(costs),
            "cost_total": sum(costs),
        }

    reasoning_cost = [
        sum(t["projected_cost_usd"] for t in r.get("telemetry", {}).values()) for r in records
    ]
    reasoning_compute = [
        sum(t["compute_s"] for t in r.get("telemetry", {}).values()) for r in records
    ]
    render_records = [r for r in records if r.get("render")]
    render_cost = [r["render"]["projected_cost_usd"] for r in render_records]
    render_compute = [r["render"]["compute_s"] for r in render_records]

    e2e_cost = [
        rc + (r["render"]["projected_cost_usd"] if r.get("render") else 0.0)
        for rc, r in zip(reasoning_cost, records)
    ]
    mean_e2e = _mean(e2e_cost)

    return {
        "n": n,
        "per_stage": per_stage,
        "reasoning": {
            "compute_s_mean": _mean(reasoning_compute),
            "cost_mean": _mean(reasoning_cost),
            "cost_total": sum(reasoning_cost),
        },
        "render": {
            "n": len(render_records),
            "compute_s_mean": _mean(render_compute),
            "cost_mean": _mean(render_cost),
            "cost_total": sum(render_cost),
        },
        "e2e": {
            "cost_mean": mean_e2e,
            "cost_total": sum(e2e_cost),
            "projected_1k": mean_e2e * 1_000,
            "projected_1m": mean_e2e * 1_000_000,
        },
    }


def format_report(summary: dict) -> str:
    """Informe legible (IDEA.md §8/§9)."""
    lines = [f"V-CoT — {summary['n']} trazas", ""]
    lines.append(f"{'etapa':<16}{'compute p50':>12}{'p90':>9}{'tok':>7}{'$/muestra':>12}")
    for stage, s in summary["per_stage"].items():
        lines.append(
            f"{s['label']+' '+stage:<16}"
            f"{s['compute_s_p50']:>11.2f}s"
            f"{s['compute_s_p90']:>8.2f}s"
            f"{s['output_tokens_mean']:>7.0f}"
            f"{s['cost_mean']:>12.6f}"
        )
    r = summary["reasoning"]
    lines.append("")
    lines.append(f"razonamiento N1–N6:  {r['compute_s_mean']:.2f}s  ${r['cost_mean']:.6f}/muestra")
    rn = summary["render"]
    if rn["n"]:
        lines.append(f"render N7 ({rn['n']}):     {rn['compute_s_mean']:.2f}s  ${rn['cost_mean']:.6f}/muestra")
    e = summary["e2e"]
    lines.append("")
    lines.append(
        f"E2E: ${e['cost_mean']:.6f}/muestra  ·  "
        f"1k=${e['projected_1k']:.2f}  ·  1M=${e['projected_1m']:.0f}"
    )
    return "\n".join(lines)


def write_csv(summary: dict, path: str) -> None:
    """Vuelca el desglose por etapa a CSV (para los notebooks de §10)."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "label", "compute_s_mean", "compute_s_p50", "compute_s_p90", "output_tokens_mean", "cost_mean", "cost_total"])
        for stage, s in summary["per_stage"].items():
            w.writerow([
                stage, s["label"], f"{s['compute_s_mean']:.6f}", f"{s['compute_s_p50']:.6f}",
                f"{s['compute_s_p90']:.6f}", f"{s['output_tokens_mean']:.2f}",
                f"{s['cost_mean']:.8f}", f"{s['cost_total']:.8f}",
            ])


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-analyze",
        description="Informe de coste/latencia de trazas V-CoT (IDEA.md §8).",
    )
    parser.add_argument("jsonl", help="Ruta a traces.jsonl")
    parser.add_argument("--csv", default=None, help="Escribir el desglose por etapa a este CSV.")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    summary = summarize(load_jsonl(args.jsonl))
    print(format_report(summary))
    if args.csv:
        write_csv(summary, args.csv)
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
