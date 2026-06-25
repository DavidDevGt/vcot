"""Tests del código puro añadido en la auditoría P0–P2."""

import pytest

from vcot.dataset import derive_seed, prompt_stratum, SEED_PROMPTS
from vcot.eval import duplicate_indices, hamming, layout_faithfulness, phash_dct
from vcot.eval.calibration import correlate, spearman, suggest_threshold
from vcot.eval.dedup import _dct_1d
from vcot.eval.safety import classify_safety, nsfw_label
from vcot.eval.stats import by_stratum, distribution, percentile, quality_report


# --------------------------------------------------------------------------- #
# P0-1 seeds
# --------------------------------------------------------------------------- #


def test_derive_seed_deterministic_and_bounded():
    a = derive_seed("a lone astronaut")
    assert a == derive_seed("a lone astronaut")  # determinista
    assert derive_seed("otro prompt") != a       # distinto por clave
    assert 0 <= a < 2**32


# --------------------------------------------------------------------------- #
# P1-2b stratum classification
# --------------------------------------------------------------------------- #


def test_prompt_stratum_curated_and_generated():
    assert prompt_stratum(SEED_PROMPTS[0]) == "portrait"   # primer bloque curado
    assert prompt_stratum(SEED_PROMPTS[3]) == "landscape"
    assert prompt_stratum("a weathered lighthouse keeper during a citywide power outage") == "portrait"
    assert prompt_stratum("xyzzy nonsense subject") == "unknown"


# --------------------------------------------------------------------------- #
# P0-2 faithfulness coverage
# --------------------------------------------------------------------------- #


def test_faithfulness_separates_coverage_from_placement():
    entities = [
        {"label": "astronaut", "bbox": (0.3, 0.4, 0.6, 0.9)},
        {"label": "pillar", "bbox": (0.7, 0.2, 0.9, 0.95)},
    ]
    # astronaut: detectado pero MAL ubicado; pillar: no detectado.
    detections = [{"label": "astronaut", "bbox": (0.0, 0.0, 0.2, 0.2), "score": 0.8}]
    res = layout_faithfulness(entities, detections, iou_threshold=0.3)
    assert res["detection_coverage"] == 0.5  # 1/2 detectado
    assert res["score"] == 0.0               # 0/2 bien ubicado
    assert res["note"]


# --------------------------------------------------------------------------- #
# P1-2a pHash DCT
# --------------------------------------------------------------------------- #


def test_dct_1d_matches_reference():
    # DCT-II de [1,1,1,1]: solo el término DC es no nulo.
    out = _dct_1d([1.0, 1.0, 1.0, 1.0])
    assert out[0] == pytest.approx(4.0)
    assert all(abs(v) < 1e-9 for v in out[1:])


def test_phash_dct_on_synthetic_images():
    Image = pytest.importorskip("PIL.Image")  # Pillow vive en el contenedor de eval

    flat = Image.new("L", (32, 32), color=128)
    half = Image.new("L", (32, 32), color=128)
    for x in range(16):
        for y in range(32):
            half.putpixel((x, y), 0)  # mitad negra → estructura distinta
    h_flat, h_half = phash_dct(flat), phash_dct(half)
    assert hamming(h_flat, h_flat) == 0
    assert hamming(h_flat, h_half) > 0  # imágenes estructuralmente distintas


# --------------------------------------------------------------------------- #
# P1-2b stats
# --------------------------------------------------------------------------- #


def test_percentile_and_distribution():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    d = distribution([1.0, 2.0, 3.0, None, 4.0])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["max"] == 4.0


def _rec(stratum, clip, passed):
    return {"dataset": {"stratum": stratum,
                        "quality": {"clip_score": clip, "passed_gate": passed}}}


def test_by_stratum_groups_and_rates():
    records = [_rec("portrait", 0.3, True), _rec("portrait", 0.1, False),
               _rec("abstract", 0.2, True)]
    bs = by_stratum(records)
    assert bs["portrait"]["n"] == 2
    assert bs["portrait"]["gate_pass_rate"] == 0.5
    assert bs["abstract"]["means"]["clip_score"] == 0.2


def test_quality_report_shape():
    rep = quality_report([_rec("portrait", 0.3, True), _rec("abstract", 0.2, False)])
    assert rep["gate"] == {"passed": 1, "failed": 1, "unknown": 0}
    assert "clip_score" in rep["distributions"]
    assert set(rep["by_stratum"]) == {"portrait", "abstract"}


# --------------------------------------------------------------------------- #
# P1-1 calibration
# --------------------------------------------------------------------------- #


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_suggest_threshold_separable():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    res = suggest_threshold(scores, labels)
    assert res["f1"] == pytest.approx(1.0)
    assert 0.2 < res["threshold"] <= 0.8


def test_correlate_per_metric():
    rows = [
        {"clip_score": 0.1, "human_good": 0},
        {"clip_score": 0.2, "human_good": 0},
        {"clip_score": 0.8, "human_good": 1},
        {"clip_score": 0.9, "human_good": 1},
    ]
    rep = correlate(rows, metrics=["clip_score"])
    assert rep["clip_score"]["n"] == 4
    # Empates en human_good (dos 0, dos 1) → Spearman con rangos promedio ≈ 0.894.
    assert rep["clip_score"]["spearman"] > 0.85


# --------------------------------------------------------------------------- #
# P2-2 safety
# --------------------------------------------------------------------------- #


def test_nsfw_label_bands():
    assert nsfw_label(0.01) == "ok"
    assert nsfw_label(0.6) == "review"
    assert nsfw_label(0.9) == "blocked"
    assert nsfw_label(None) == "unknown"


def test_classify_safety_structure_and_release():
    safe = classify_safety(0.01)
    assert safe["nsfw_label"] == "ok" and safe["release_blocked"] is False
    assert "csam_hash" in safe["checks_pending"]  # honesto sobre lo no implementado
    blocked = classify_safety(0.95)
    assert blocked["release_blocked"] is True
