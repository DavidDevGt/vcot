"""Estadística de calidad del dataset: distribuciones + breakdown por estrato.

Un eval de nivel lab no reporta solo *medias* (esconden la cola). Aquí calculamos
**percentiles** (p10/p50/p90) por métrica y un **desglose por estrato** (¿la
faithfulness es peor en 'abstracto' que en 'retrato'? casi siempre sí). Puro
(stdlib), sobre los records ya evaluados (con bloque ``dataset.quality``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

#: Métricas continuas que se resumen con distribución.
METRIC_KEYS = ("clip_score", "aesthetic", "image_reward", "faithfulness", "detection_coverage")


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Percentil ``p`` (0–100) con interpolación lineal. ``[]`` → ``None``."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (idx - lo))


def distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """``{n, mean, p10, p50, p90, min, max}`` de una métrica (ignora ``None``)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None,
                "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 4),
        "p10": round(percentile(vals, 10), 4),
        "p50": round(percentile(vals, 50), 4),
        "p90": round(percentile(vals, 90), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def _quality(record: Dict) -> Dict:
    return (record.get("dataset") or {}).get("quality") or {}


def metric_distributions(records: Sequence[Dict]) -> Dict[str, Dict]:
    """Distribución de cada métrica continua sobre el dataset."""
    return {
        key: distribution([_quality(r).get(key) for r in records])
        for key in METRIC_KEYS
    }


def by_stratum(records: Sequence[Dict]) -> Dict[str, Dict]:
    """Breakdown por estrato: conteo, tasa de gate y medias de métricas."""
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        stratum = (r.get("dataset") or {}).get("stratum") or "unknown"
        groups.setdefault(stratum, []).append(r)

    out: Dict[str, Dict] = {}
    for stratum, recs in sorted(groups.items()):
        q = [_quality(r) for r in recs]
        passed = sum(1 for x in q if x.get("passed_gate"))
        out[stratum] = {
            "n": len(recs),
            "gate_pass_rate": round(passed / len(recs), 4) if recs else None,
            "means": {
                key: distribution([x.get(key) for x in q])["mean"]
                for key in METRIC_KEYS
            },
        }
    return out


def quality_report(records: Sequence[Dict]) -> Dict:
    """Resumen completo de calidad (distribuciones + estratos + gate global)."""
    q = [_quality(r) for r in records]
    passed = sum(1 for x in q if x.get("passed_gate"))
    failed = sum(1 for x in q if x.get("passed_gate") is False)
    return {
        "n": len(records),
        "gate": {"passed": passed, "failed": failed,
                 "unknown": len(records) - passed - failed},
        "distributions": metric_distributions(records),
        "by_stratum": by_stratum(records),
    }


def format_quality_report(report: Dict) -> str:
    """Render legible del resumen de calidad (para stdout/informe)."""
    lines = [f"Calidad — {report['n']} muestras "
             f"(gate: {report['gate']['passed']} ok / {report['gate']['failed']} no)", ""]
    lines.append(f"{'métrica':<20}{'p10':>8}{'p50':>8}{'p90':>8}{'mean':>8}")
    for key, d in report["distributions"].items():
        if d["n"]:
            lines.append(
                f"{key:<20}{d['p10']:>8.3f}{d['p50']:>8.3f}{d['p90']:>8.3f}{d['mean']:>8.3f}"
            )
    lines.append("")
    lines.append(f"{'estrato':<16}{'n':>4}{'gate%':>7}{'clip':>7}{'aesth':>7}{'faith':>7}{'cover':>7}")
    for stratum, s in report["by_stratum"].items():
        m = s["means"]
        lines.append(
            f"{stratum:<16}{s['n']:>4}{(s['gate_pass_rate'] or 0)*100:>6.0f}%"
            f"{_fmt(m['clip_score']):>7}{_fmt(m['aesthetic']):>7}"
            f"{_fmt(m['faithfulness']):>7}{_fmt(m['detection_coverage']):>7}"
        )
    return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "  -  "
