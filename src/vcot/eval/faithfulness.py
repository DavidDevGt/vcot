"""Layout-faithfulness: ¿la imagen respeta el scene-graph? (IDEA.md §3, §8.6).

La **métrica estrella** del proyecto. Mide cuánto del layout decidido en N2
(entidades + bboxes) se cumple realmente en el píxel renderizado: un detector
open-vocab (OWLv2/GroundingDINO, en ``modal_app/eval.py``) detecta las entidades
en la imagen y aquí calculamos el emparejamiento por **IoU** contra las bboxes
del layout.

Este módulo es **puro y determinista** (sin modelos): recibe entidades y
detecciones como dicts ``{"label", "bbox"}`` con ``bbox`` normalizada
``[x0, y0, x1, y1]`` en ``[0,1]``. Así se testea sin GPU y el detector se acopla
por fuera.

Hoy el render solo usa *prompt-enrichment* (el layout no condiciona el píxel —
E2 quedó fuera de alcance), así que esta métrica **cuantifica el gap**: es la
línea base sobre la que mediría cualquier mecanismo de control futuro.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]

#: Entidades con bbox física detectable. Luz/atmósfera/sombra no tienen una caja
#: fiable que un detector pueda localizar, así que no entran en la métrica.
DETECTABLE_KINDS = frozenset({"object", "character"})


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union de dos bboxes ``[x0, y0, x1, y1]``."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detectable_entities(layout: Dict) -> List[Dict]:
    """Extrae del layout las entidades con bbox detectable (``label``+``bbox``)."""
    return [
        {"label": str(e["id"]), "bbox": tuple(e["bbox"])}
        for e in layout.get("entities", [])
        if e.get("kind") in DETECTABLE_KINDS
    ]


def _normalize_label(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def detector_query(label: str) -> str:
    """Texto de consulta para el detector open-vocab a partir de un id de entidad.

    Los ids del LLM vienen en *snake_case* (``stained_glass``); un detector de
    sustantivos comunes necesita lenguaje natural (``stained glass``). Sin esto la
    recall de detección colapsa y la métrica mide al detector, no al layout.
    """
    return _normalize_label(label)


def _label_match(entity_label: str, detection_label: str) -> bool:
    """El detector se consulta con los labels de las entidades, así que el match
    es por igualdad normalizada o inclusión (tolera 'astronaut' vs 'an astronaut')."""
    a, b = _normalize_label(entity_label), _normalize_label(detection_label)
    return a == b or a in b or b in a


#: La faithfulness es una **línea base**: el render actual solo usa
#: prompt-enrichment (el layout N2 NO condiciona el píxel — E2 fuera de alcance).
#: Por eso ``score`` mide el *gap* entre el layout decidido y el render libre, no
#: la fidelidad de un mecanismo de control. Se reporta junto a
#: ``detection_coverage`` para separar "no detectado" de "detectado pero mal
#: ubicado" — sin esa separación el número mide al detector, no al layout.
BASELINE_NOTE = (
    "baseline (prompt-only conditioning; layout no condiciona el pixel). "
    "score = colocacion correcta; detection_coverage = recall del detector."
)


def layout_faithfulness(
    entities: Sequence[Dict],
    detections: Sequence[Dict],
    *,
    iou_threshold: float = 0.3,
) -> Dict:
    """Empareja entidades↔detecciones (greedy por IoU + label) y puntúa.

    Devuelve:
      - ``score``: fracción de entidades **bien ubicadas** (label match + IoU ≥
        umbral) — la fidelidad de colocación.
      - ``detection_coverage``: fracción de entidades con **alguna** detección de
        su label (IoU cualquiera) — recall del detector; aísla el fallo de
        detección del fallo de layout.
      - ``mean_iou`` (sobre entidades con detección) y diagnóstico por entidad.

    Si no hay entidades detectables, ``score`` es ``None`` (no aplica).
    """
    ents = list(entities)
    if not ents:
        return {
            "score": None, "detection_coverage": None, "mean_iou": None,
            "n_entities": 0, "n_matched": 0, "n_detected": 0,
            "iou_threshold": iou_threshold, "per_entity": [], "note": BASELINE_NOTE,
        }

    used: set[int] = set()
    per_entity: List[Dict] = []
    for ent in ents:
        best_iou, best_j = 0.0, -1
        detected = False  # ¿el detector emitió ALGO con este label?
        for j, det in enumerate(detections):
            if not _label_match(ent["label"], det["label"]):
                continue
            detected = True
            if j in used:
                continue
            score = iou(tuple(ent["bbox"]), tuple(det["bbox"]))
            if score > best_iou:
                best_iou, best_j = score, j
        matched = best_j >= 0 and best_iou >= iou_threshold
        if matched:
            used.add(best_j)
        per_entity.append(
            {
                "label": ent["label"], "iou": round(best_iou, 4),
                "detected": detected, "matched": matched,
            }
        )

    n_matched = sum(1 for p in per_entity if p["matched"])
    n_detected = sum(1 for p in per_entity if p["detected"])
    ious_detected = [p["iou"] for p in per_entity if p["detected"]]
    mean_iou = (sum(ious_detected) / len(ious_detected)) if ious_detected else 0.0
    return {
        "score": round(n_matched / len(ents), 4),
        "detection_coverage": round(n_detected / len(ents), 4),
        "mean_iou": round(mean_iou, 4),
        "n_entities": len(ents),
        "n_matched": n_matched,
        "n_detected": n_detected,
        "iou_threshold": iou_threshold,
        "per_entity": per_entity,
        "note": BASELINE_NOTE,
    }
