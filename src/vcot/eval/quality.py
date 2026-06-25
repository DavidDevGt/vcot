"""Agregación de eval → bloque ``dataset`` + gate de calidad (IDEA.md §4.2).

Reúne los scores por muestra (CLIP, aesthetic, faithfulness, NSFW, dedup),
decide si la muestra **pasa el gate** de calidad y los fusiona en el bloque
``dataset`` de la traza (``quality``/``safety``/``split``). Puro y testeable; los
scores los producen los modelos de ``modal_app/eval.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

#: Umbrales por defecto del gate. ``None`` en un score ⇒ la dimensión no aplica
#: (p.ej. faithfulness sin entidades detectables) y no penaliza.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "clip_min": 0.22,          # cosine prompt↔imagen (CLIP ViT-L/14)
    "aesthetic_min": 4.5,      # predictor LAION (escala ~1–10)
    "faithfulness_min": 0.3,   # fracción de entidades respetadas
    "nsfw_max": 0.5,           # prob. de contenido NSFW
}


def passes_gate(
    quality: Dict[str, Optional[float]],
    safety: Dict[str, Optional[float]],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    is_duplicate: bool = False,
) -> Tuple[bool, List[str]]:
    """¿La muestra pasa el gate? Devuelve ``(passed, reasons)``."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: List[str] = []

    if is_duplicate:
        reasons.append("duplicate")

    clip = quality.get("clip_score")
    if clip is not None and clip < th["clip_min"]:
        reasons.append("low_clip")

    aesthetic = quality.get("aesthetic")
    if aesthetic is not None and aesthetic < th["aesthetic_min"]:
        reasons.append("low_aesthetic")

    faithfulness = quality.get("faithfulness")
    if faithfulness is not None and faithfulness < th["faithfulness_min"]:
        reasons.append("low_faithfulness")

    nsfw = safety.get("nsfw")
    if nsfw is not None and nsfw > th["nsfw_max"]:
        reasons.append("nsfw")
    if safety.get("release_blocked"):
        reasons.append("release_blocked")

    return (len(reasons) == 0, reasons)


def merge_eval(
    record: Dict,
    *,
    quality: Dict[str, Optional[float]],
    safety: Dict[str, Optional[float]],
    split: Optional[str],
    is_duplicate: bool = False,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Fusiona los scores en ``record['dataset']`` (in-place) y devuelve el record."""
    passed, reasons = passes_gate(
        quality, safety, thresholds=thresholds, is_duplicate=is_duplicate
    )
    ds = dict(record.get("dataset") or {})
    ds.setdefault("license", None)
    ds.setdefault("code_version", None)
    ds["split"] = split
    ds["quality"] = {**quality, "passed_gate": passed, "gate_reasons": reasons}
    ds["safety"] = {**safety, "is_duplicate": is_duplicate}
    record["dataset"] = ds
    return record
