"""Validación humana + calibración de umbrales (IDEA.md §8, metodología).

Sin validar las métricas automáticas contra juicio humano, los umbrales del gate
(``vcot.eval.quality.DEFAULT_THRESHOLDS``) son arbitrarios — y un reviewer lo
señalaría. Este módulo cierra ese hueco:

1. ``emit_sheet`` muestrea N muestras y escribe una **planilla de etiquetado**
   (CSV: id, prompt, imagen, scores automáticos + columna ``human_good`` vacía).
2. Un humano la completa (1 = buena, 0 = mala).
3. ``calibrate`` ingiere la planilla y reporta la **correlación de Spearman** de
   cada métrica con el juicio humano y un **umbral sugerido** (máx F1) por métrica.

La matemática (Spearman, sugerencia de umbral) es pura y testeable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from vcot.eval.stats import METRIC_KEYS


# --------------------------------------------------------------------------- #
# Matemática pura
# --------------------------------------------------------------------------- #


def _ranks(values: Sequence[float]) -> List[float]:
    """Rangos promedio (maneja empates) — base de Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # rangos 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Correlación de rango de Spearman ∈ [-1, 1]; ``None`` si no es computable."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    r = _pearson(_ranks(xs), _ranks(ys))
    return round(r, 4) if r is not None else None


def suggest_threshold(
    scores: Sequence[float], labels: Sequence[int]
) -> Dict[str, Optional[float]]:
    """Umbral (``score ≥ t``) que maximiza F1 contra etiquetas humanas binarias."""
    pairs = [(s, int(l)) for s, l in zip(scores, labels) if s is not None and l is not None]
    if not pairs or all(l == 0 for _, l in pairs) or all(l == 1 for _, l in pairs):
        return {"threshold": None, "precision": None, "recall": None, "f1": None}

    best = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for cand, _ in sorted(set(pairs)):
        tp = sum(1 for s, l in pairs if s >= cand and l == 1)
        fp = sum(1 for s, l in pairs if s >= cand and l == 0)
        fn = sum(1 for s, l in pairs if s < cand and l == 1)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best["f1"]:
            best = {"threshold": round(cand, 4), "precision": round(precision, 4),
                    "recall": round(recall, 4), "f1": round(f1, 4)}
    return best


def correlate(
    rows: Sequence[Dict], *, metrics: Sequence[str] = METRIC_KEYS
) -> Dict[str, Dict]:
    """Por métrica: Spearman vs ``human_good`` + umbral sugerido.

    ``rows`` = dicts con las métricas y ``human_good`` (0/1). Devuelve un reporte
    de calibración por métrica.
    """
    human = [r.get("human_good") for r in rows]
    out: Dict[str, Dict] = {}
    for metric in metrics:
        vals = [r.get(metric) for r in rows]
        paired = [(v, h) for v, h in zip(vals, human) if v is not None and h is not None]
        if not paired:
            out[metric] = {"n": 0, "spearman": None, "suggested": None}
            continue
        v, h = zip(*paired)
        out[metric] = {
            "n": len(paired),
            "spearman": spearman(v, h),
            "suggested": suggest_threshold(v, h),
        }
    return out


# --------------------------------------------------------------------------- #
# IO: planilla de etiquetado / ingesta
# --------------------------------------------------------------------------- #

_SHEET_COLS = ["id", "prompt", "image", "human_good", *METRIC_KEYS]


def emit_sheet(
    eval_jsonl: str, out_csv: str, *, n: int = 100, images: str = "outputs", seed: int = 0
) -> int:
    """Muestrea ``n`` muestras → CSV de etiquetado (``human_good`` en blanco)."""
    with open(eval_jsonl, encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()]
    rng = random.Random(seed)
    rng.shuffle(records)
    sample = records[:n]
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_SHEET_COLS)
        w.writeheader()
        for r in sample:
            q = (r.get("dataset") or {}).get("quality") or {}
            row = {"id": r["id"], "prompt": r["prompt"],
                   "image": f"{images}/{r['id']}_0.webp", "human_good": ""}
            for key in METRIC_KEYS:
                row[key] = q.get(key)
            w.writerow(row)
    return len(sample)


def load_labeled_sheet(csv_path: str) -> List[Dict]:
    """Lee la planilla completada → rows con ``human_good`` int y métricas float."""
    rows: List[Dict] = []
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            if raw.get("human_good", "").strip() == "":
                continue  # sin etiquetar → se ignora
            row: Dict = {"id": raw["id"], "human_good": int(float(raw["human_good"]))}
            for key in METRIC_KEYS:
                val = raw.get(key, "").strip()
                row[key] = float(val) if val not in ("", "None") else None
            rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-calibrate", description="Validación humana + calibración de umbrales."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sheet", help="Emitir planilla de etiquetado.")
    s.add_argument("eval_jsonl")
    s.add_argument("--out", default="outputs/label_sheet.csv")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--images", default="outputs")
    s.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("calibrate", help="Calibrar desde una planilla etiquetada.")
    c.add_argument("labeled_csv")

    args = parser.parse_args(argv)
    if args.cmd == "sheet":
        n = emit_sheet(args.eval_jsonl, args.out, n=args.n, images=args.images, seed=args.seed)
        print(f"{n} muestras -> {args.out} (completá la columna human_good con 1/0)")
    else:
        rows = load_labeled_sheet(args.labeled_csv)
        report = correlate(rows)
        print(f"Calibración sobre {len(rows)} etiquetas humanas:\n")
        print(f"{'métrica':<20}{'n':>5}{'spearman':>10}{'umbral':>9}{'F1':>7}")
        for metric, r in report.items():
            sug = r.get("suggested") or {}
            print(
                f"{metric:<20}{r['n']:>5}"
                f"{_fmt(r['spearman']):>10}{_fmt(sug.get('threshold')):>9}{_fmt(sug.get('f1')):>7}"
            )


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


if __name__ == "__main__":
    main()
