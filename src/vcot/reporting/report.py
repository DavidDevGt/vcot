"""Informe final de inferencia y coste del pipeline V-CoT (IDEA.md §8, §9).

Toma las trazas (``traces.jsonl``) y el ledger de ejecuciones (``runs.jsonl``) y
produce un **bundle con fecha** en ``reports/<timestamp>/``: Markdown (humano),
JSON (máquina) y CSV (por etapa). Pensado para correr *después* de los procesos,
de modo que quede todo registrado de forma profesional.

    python -m vcot.reporting outputs/ --out reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Sequence

import vcot
from vcot.analysis.aggregate import _pct, load_jsonl, summarize, write_csv
from vcot.pipeline.schemas import STAGE_LABELS, STAGE_MODELS
from vcot.reporting.runlog import load_runs
from vcot.telemetry.rates import gpu_rate


def _record_e2e(r: dict) -> tuple[float, float]:
    """(compute_s, cost_usd) extremo a extremo de un registro (razonamiento + render)."""
    tele = r.get("telemetry", {})
    compute = sum(t["compute_s"] for t in tele.values())
    cost = sum(t["projected_cost_usd"] for t in tele.values())
    if r.get("render"):
        compute += r["render"]["compute_s"]
        cost += r["render"]["projected_cost_usd"]
    return compute, cost


def build_report(
    records: List[dict],
    runs: Optional[List[dict]] = None,
    container_costs: Optional[List[dict]] = None,
) -> dict:
    """Construye el informe estructurado a partir de trazas (+ ledger opcional).

    ``container_costs`` son los registros de :class:`ContainerMeter` (uno por
    contenedor Modal): el **coste real facturado** (carga + inferencias + idle +
    CPU/mem), no el marginal. Si se pasan, el informe reporta el gasto real medido.
    """
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = len(records)
    if n == 0:
        return {"generated_at": generated_at, "n_traces": 0, "vcot_version": vcot.__version__}

    core = summarize(records)

    e2e = [_record_e2e(r) for r in records]
    e2e_compute = [c for c, _ in e2e]
    e2e_cost = [c for _, c in e2e]

    # Calidad: reintentos por etapa y validez a la primera.
    retries_by_stage = {
        stage: sum(r["telemetry"][stage].get("retries", 0) for r in records if stage in r.get("telemetry", {}))
        for stage in STAGE_MODELS
    }
    total_retries = sum(retries_by_stage.values())
    first_try = sum(
        1 for r in records if sum(t.get("retries", 0) for t in r.get("telemetry", {}).values()) == 0
    )

    tokens_total = sum(
        t.get("output_tokens", 0) for r in records for t in r.get("telemetry", {}).values()
    )

    # Recursos.
    planner_gpus = Counter(
        t["projected_gpu"] for r in records for t in r.get("telemetry", {}).values()
    )
    render_gpus = Counter(r["render"]["projected_gpu"] for r in records if r.get("render"))
    planner_models = Counter(str(r.get("meta", {}).get("planner", "unknown")) for r in records)

    dates = sorted(r["meta"]["created_at"] for r in records if r.get("meta", {}).get("created_at"))
    date_range = {"from": dates[0], "to": dates[-1]} if dates else None

    used_gpus = set(planner_gpus) | set(render_gpus)
    rates = {}
    for g in used_gpus:
        try:
            rates[g] = gpu_rate(g)
        except KeyError:
            rates[g] = None

    mean_e2e = sum(e2e_cost) / n
    report = {
        "generated_at": generated_at,
        "vcot_version": vcot.__version__,
        "n_traces": n,
        "n_with_image": sum(1 for r in records if r.get("final_image")),
        "date_range": date_range,
        "models": {"planner": dict(planner_models)},
        "gpus": {"planner": dict(planner_gpus), "render": dict(render_gpus)},
        "rates_usd_per_s": rates,
        "cost": core,
        "total_dataset_cost_usd": round(sum(e2e_cost), 8),
        "cost_projection": {
            "1k": mean_e2e * 1_000,
            "10k": mean_e2e * 10_000,
            "100k": mean_e2e * 100_000,
            "1m": mean_e2e * 1_000_000,
        },
        "latency": {
            "e2e_mean_s": sum(e2e_compute) / n,
            "e2e_p50_s": _pct(e2e_compute, 50),
            "e2e_p90_s": _pct(e2e_compute, 90),
            "e2e_p99_s": _pct(e2e_compute, 99),
        },
        "tokens_total": tokens_total,
        "quality": {
            "total_retries": total_retries,
            "retries_by_stage": retries_by_stage,
            "first_try_valid_rate": first_try / n,
        },
    }
    if runs:
        report["ledger"] = {
            "n_runs": len(runs),
            "total_cost_usd": round(sum(float(r.get("total_cost_usd", 0.0)) for r in runs), 6),
            "total_real_cost_est_usd": round(
                sum(float(r.get("real_cost_est_usd", 0.0)) for r in runs), 6
            ),
            "by_kind": dict(Counter(r.get("kind", "?") for r in runs)),
            "runs": runs,
        }
    if container_costs:
        real_total = sum(float(c.get("real_cost_usd", 0.0)) for c in container_costs)
        marginal_total = report.get("total_dataset_cost_usd", 0.0)
        report["real_billed"] = {
            "n_containers": len(container_costs),
            "total_real_cost_usd": round(real_total, 6),
            "total_marginal_cost_usd": round(marginal_total, 6),
            # Cuánto subestima el coste marginal al gasto real (×). Para corridas
            # dispersas suele ser grande (carga + idle dominan); para batches densos →1.
            "real_to_marginal_ratio": round(real_total / marginal_total, 2)
            if marginal_total
            else None,
            "gpu_cost_usd": round(sum(float(c.get("gpu_cost_usd", 0.0)) for c in container_costs), 6),
            "cpu_cost_usd": round(sum(float(c.get("cpu_cost_usd", 0.0)) for c in container_costs), 6),
            "mem_cost_usd": round(sum(float(c.get("mem_cost_usd", 0.0)) for c in container_costs), 6),
            "billed_s_total": round(sum(float(c.get("billed_s", 0.0)) for c in container_costs), 1),
            "by_kind": dict(Counter(c.get("kind", "?") for c in container_costs)),
        }
    return report


def render_markdown(report: dict) -> str:
    """Renderiza el informe a Markdown profesional."""
    if report.get("n_traces", 0) == 0:
        return "# V-CoT — Informe\n\n_No hay trazas para reportar._\n"

    c = report["cost"]
    lat = report["latency"]
    q = report["quality"]
    pr = report["cost_projection"]
    dr = report.get("date_range")

    L: List[str] = []
    L.append("# V-CoT — Informe de inferencia y coste")
    L.append("")
    L.append(
        f"**Generado:** {report['generated_at']}  ·  "
        f"**Trazas:** {report['n_traces']} ({report['n_with_image']} con imagen final)  ·  "
        f"**vcot:** v{report['vcot_version']}"
    )
    planner_models = ", ".join(report["models"]["planner"]) or "—"
    gpus = ", ".join(sorted(set(report["gpus"]["planner"]) | set(report["gpus"]["render"]))) or "—"
    L.append(f"**Planner:** {planner_models}  ·  **GPUs:** {gpus}")
    if dr:
        L.append(f"**Rango de fechas:** {dr['from']} → {dr['to']}")
    L.append("")

    # 1. Resumen ejecutivo
    L.append("## 1. Resumen ejecutivo")
    L.append("")
    L.append(f"- **Coste total del dataset (N1–N7):** ${report['total_dataset_cost_usd']:.4f}")
    L.append(f"- **Coste medio/traza — razonamiento N1–N6:** ${c['reasoning']['cost_mean']:.6f}")
    L.append(f"- **Coste medio/traza — e2e N1–N7:** ${c['e2e']['cost_mean']:.6f}")
    L.append(
        f"- **Proyección e2e:** 1k=${pr['1k']:.2f} · 10k=${pr['10k']:.2f} · "
        f"100k=${pr['100k']:.0f} · 1M=${pr['1m']:.0f}"
    )
    L.append(
        f"- **Latencia e2e:** media {lat['e2e_mean_s']:.2f}s · "
        f"p50 {lat['e2e_p50_s']:.2f}s · p90 {lat['e2e_p90_s']:.2f}s · p99 {lat['e2e_p99_s']:.2f}s"
    )
    L.append(f"- **Tokens generados (total):** {report['tokens_total']:,}")
    L.append(
        f"- **Validez a la primera:** {q['first_try_valid_rate']:.0%}  ·  "
        f"reintentos totales: {q['total_retries']}"
    )
    rb = report.get("real_billed")
    if rb:
        ratio = rb["real_to_marginal_ratio"]
        ratio_txt = f" (**{ratio}× el marginal**)" if ratio else ""
        L.append(
            f"- **⚠️ Coste REAL facturado por Modal:** "
            f"${rb['total_real_cost_usd']:.4f}{ratio_txt} — "
            f"vs. ${rb['total_marginal_cost_usd']:.4f} marginal. "
            f"Incluye carga de modelo + idle del scaledown + CPU/mem "
            f"({rb['n_containers']} contenedores, {rb['billed_s_total']:.0f}s facturados)"
        )
    L.append("")

    # 2. Coste/latencia por etapa
    L.append("## 2. Coste y latencia por etapa de razonamiento (N1–N6)")
    L.append("")
    L.append("| Etapa | compute p50 | p90 | tokens | $/muestra | $ total |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for stage, s in c["per_stage"].items():
        L.append(
            f"| {s['label']} {stage} | {s['compute_s_p50']:.2f}s | {s['compute_s_p90']:.2f}s | "
            f"{s['output_tokens_mean']:.0f} | ${s['cost_mean']:.6f} | ${s['cost_total']:.6f} |"
        )
    L.append("")

    # 3. Render
    L.append("## 3. Render (N7)")
    L.append("")
    rn = c["render"]
    if rn["n"]:
        L.append(
            f"- Imágenes: {rn['n']}  ·  compute medio: {rn['compute_s_mean']:.2f}s  ·  "
            f"$/imagen: ${rn['cost_mean']:.6f}  ·  $ total: ${rn['cost_total']:.6f}"
        )
    else:
        L.append("_Sin renders en este dataset (solo razonamiento N1–N6)._")
    L.append("")

    # 4. Calidad
    L.append("## 4. Calidad del razonamiento")
    L.append("")
    L.append("| Etapa | reintentos |")
    L.append("|---|--:|")
    for stage, rr in q["retries_by_stage"].items():
        L.append(f"| {STAGE_LABELS[stage]} {stage} | {rr} |")
    L.append("")

    # 5. Recursos y reproducibilidad
    L.append("## 5. Recursos y reproducibilidad")
    L.append("")
    L.append("| GPU | $/segundo |")
    L.append("|---|--:|")
    for g, rate in sorted(report["rates_usd_per_s"].items()):
        L.append(f"| {g} | {rate if rate is None else f'{rate:.6f}'} |")
    L.append("")
    L.append("Tarifas tomadas de `vcot.telemetry.rates` (única fuente de verdad, IDEA.md §8.1).")
    L.append("")

    # 6. Coste real facturado
    L.append("## 6. Coste real facturado por Modal")
    L.append("")
    rb = report.get("real_billed")
    if rb:
        L.append(
            "Medido por `ContainerMeter` (uno por contenedor): vida completa "
            "(carga + inferencias + idle del `scaledown_window`) × tarifa GPU+CPU+memoria. "
            "Es lo que **realmente pagás**, no el coste marginal de inferencia."
        )
        L.append("")
        L.append("| Concepto | USD |")
        L.append("|---|--:|")
        L.append(f"| **Coste real total** | **${rb['total_real_cost_usd']:.4f}** |")
        L.append(f"| Coste marginal (inferencia) | ${rb['total_marginal_cost_usd']:.4f} |")
        if rb["real_to_marginal_ratio"]:
            L.append(f"| Ratio real/marginal | {rb['real_to_marginal_ratio']}× |")
        L.append(f"| — del cual GPU | ${rb['gpu_cost_usd']:.4f} |")
        L.append(f"| — del cual CPU | ${rb['cpu_cost_usd']:.4f} |")
        L.append(f"| — del cual memoria | ${rb['mem_cost_usd']:.4f} |")
        L.append(f"| Contenedores facturados | {rb['n_containers']} ({rb['billed_s_total']:.0f}s) |")
        L.append("")
    else:
        L.append(
            "_Sin `container_costs.jsonl` (lo escribe `@modal.exit` de cada app GPU). "
            "Mientras tanto, el ledger trae el coste real **estimado** por corrida._"
        )
        L.append("")

    # 7. Ledger
    L.append("## 7. Registro de ejecuciones")
    L.append("")
    ledger = report.get("ledger")
    if ledger:
        L.append(
            f"{ledger['n_runs']} ejecuciones registradas  ·  "
            f"coste marginal: ${ledger['total_cost_usd']:.4f}  ·  "
            f"coste real estimado: ${ledger.get('total_real_cost_est_usd', 0.0):.4f}  ·  "
            f"por tipo: {ledger['by_kind']}"
        )
        L.append("")
        L.append("| Fecha | Tipo | Modelo | GPU | Ítems | Marginal | Real est | Dur | Estado |")
        L.append("|---|---|---|---|--:|--:|--:|--:|---|")
        for r in ledger["runs"][-20:]:
            L.append(
                f"| {r.get('started_at','')} | {r.get('kind','')} | {r.get('model') or '—'} | "
                f"{r.get('gpu') or '—'} | {r.get('n_items',0)} | "
                f"${float(r.get('total_cost_usd',0)):.4f} | "
                f"${float(r.get('real_cost_est_usd',0)):.4f} | {r.get('duration_s',0):.1f}s | "
                f"{r.get('status','')} |"
            )
    else:
        L.append("_Sin ledger de ejecuciones (`runs.jsonl` no encontrado)._")
    L.append("")
    L.append("---")
    L.append("*Generado por `vcot.reporting`.*")
    return "\n".join(L)


def write_bundle(
    report: dict,
    out_dir: str,
    records: List[dict],
    runs: Optional[List[dict]] = None,
) -> dict:
    """Escribe el bundle del informe (md/json/csv) en ``out_dir/<timestamp>/``."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = os.path.join(out_dir, ts)
    os.makedirs(bundle, exist_ok=True)

    md = render_markdown(report)
    md_path = os.path.join(bundle, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    json_path = os.path.join(bundle, "report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    paths = {"dir": bundle, "markdown": md_path, "json": json_path}

    if report.get("n_traces", 0) > 0:
        csv_path = os.path.join(bundle, "cost_by_stage.csv")
        write_csv(report["cost"], csv_path)
        paths["cost_csv"] = csv_path

    if runs:
        runs_csv = os.path.join(bundle, "runs.csv")
        with open(runs_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "started_at", "kind", "model", "gpu", "n_items",
                "total_cost_usd", "real_cost_est_usd", "duration_s", "status",
            ])
            for r in runs:
                w.writerow([
                    r.get("started_at", ""), r.get("kind", ""), r.get("model", ""),
                    r.get("gpu", ""), r.get("n_items", 0), r.get("total_cost_usd", 0.0),
                    r.get("real_cost_est_usd", 0.0), r.get("duration_s", 0.0), r.get("status", ""),
                ])
        paths["runs_csv"] = runs_csv

    # Copia de conveniencia siempre actualizada.
    latest = os.path.join(out_dir, "latest.md")
    shutil.copyfile(md_path, latest)
    paths["latest"] = latest
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-report",
        description="Genera el informe final del pipeline V-CoT (IDEA.md §8/§9).",
    )
    parser.add_argument("indir", nargs="?", default="outputs", help="Carpeta con traces.jsonl / runs.jsonl")
    parser.add_argument("--out", default="reports", help="Carpeta donde escribir el bundle.")
    parser.add_argument("--traces", default=None, help="Ruta explícita a traces.jsonl.")
    parser.add_argument("--runs", default=None, help="Ruta explícita a runs.jsonl.")
    args = parser.parse_args(argv)

    # El informe usa caracteres UTF-8 (→, ·); la consola de Windows es cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    traces_path = args.traces or os.path.join(args.indir, "traces.jsonl")
    runs_path = args.runs or os.path.join(args.indir, "runs.jsonl")
    container_costs_path = os.path.join(args.indir, "container_costs.jsonl")

    if not os.path.exists(traces_path):
        raise SystemExit(f"No encuentro trazas en {traces_path}. ¿Corriste el planner/dataset?")

    records = load_jsonl(traces_path)
    runs = load_runs(runs_path)
    # Coste real medido por contenedor (lo escribe `@modal.exit`); opcional.
    container_costs = load_runs(container_costs_path)
    report = build_report(records, runs, container_costs)
    paths = write_bundle(report, args.out, records, runs)

    print(render_markdown(report))
    print("\n" + "=" * 60)
    print(f"Bundle del informe -> {paths['dir']}")
    for k, v in paths.items():
        if k != "dir":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
