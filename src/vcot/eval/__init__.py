"""Capa de evaluación research-grade del dataset V-CoT (IDEA.md §4, §8).

Núcleo **puro y testeable** (sin modelos): la matemática de la métrica estrella
(layout-faithfulness por IoU), el particionado sin fuga, el dedup perceptual y el
gate de calidad. Los modelos pesados (CLIP, OWLv2, NSFW) viven en
``modal_app/eval.py`` y se apoyan en estas funciones.
"""

from __future__ import annotations

from vcot.eval.dedup import average_hash, duplicate_indices, hamming, phash_dct
from vcot.eval.faithfulness import (
    BASELINE_NOTE,
    DETECTABLE_KINDS,
    detectable_entities,
    detector_query,
    iou,
    layout_faithfulness,
)
from vcot.eval.quality import DEFAULT_THRESHOLDS, merge_eval, passes_gate
from vcot.eval.splits import assign_split, assign_splits, split_fraction

__all__ = [
    "iou",
    "layout_faithfulness",
    "detectable_entities",
    "detector_query",
    "DETECTABLE_KINDS",
    "BASELINE_NOTE",
    "assign_split",
    "assign_splits",
    "split_fraction",
    "hamming",
    "average_hash",
    "phash_dct",
    "duplicate_indices",
    "passes_gate",
    "merge_eval",
    "DEFAULT_THRESHOLDS",
]
