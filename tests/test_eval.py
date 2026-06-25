import pytest

from vcot.eval import (
    assign_split,
    assign_splits,
    detectable_entities,
    duplicate_indices,
    hamming,
    iou,
    layout_faithfulness,
    merge_eval,
    passes_gate,
)


# --------------------------------------------------------------------------- #
# faithfulness
# --------------------------------------------------------------------------- #


def test_iou_basics():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert iou((0, 0, 0.5, 1), (0.5, 0, 1, 1)) == 0.0  # disjuntas (tocan en el borde)
    # Mitad solapada: inter 0.25, union 0.75 → 1/3.
    assert iou((0, 0, 0.5, 1), (0.25, 0, 0.75, 1)) == pytest.approx(1 / 3)


def test_detectable_entities_filters_effects():
    layout = {
        "entities": [
            {"id": "astronaut", "kind": "character", "bbox": [0.3, 0.4, 0.6, 0.9]},
            {"id": "moonlight", "kind": "light", "bbox": [0.0, 0.0, 0.3, 0.3]},
            {"id": "pillar", "kind": "object", "bbox": [0.7, 0.2, 0.9, 0.95]},
        ]
    }
    labels = [e["label"] for e in detectable_entities(layout)]
    assert labels == ["astronaut", "pillar"]  # la luz no es detectable


def test_layout_faithfulness_perfect_and_partial():
    entities = [
        {"label": "astronaut", "bbox": (0.3, 0.4, 0.6, 0.9)},
        {"label": "pillar", "bbox": (0.7, 0.2, 0.9, 0.95)},
    ]
    # Una detección casa perfecto, la otra no aparece.
    detections = [{"label": "astronaut", "bbox": (0.3, 0.4, 0.6, 0.9), "score": 0.9}]
    res = layout_faithfulness(entities, detections, iou_threshold=0.3)
    assert res["n_entities"] == 2
    assert res["n_matched"] == 1
    assert res["score"] == pytest.approx(0.5)


def test_layout_faithfulness_no_entities_is_none():
    res = layout_faithfulness([], [{"label": "x", "bbox": (0, 0, 1, 1)}])
    assert res["score"] is None


def test_layout_faithfulness_label_inclusion_match():
    entities = [{"label": "astronaut", "bbox": (0.3, 0.4, 0.6, 0.9)}]
    detections = [{"label": "an astronaut", "bbox": (0.31, 0.41, 0.61, 0.91)}]
    res = layout_faithfulness(entities, detections)
    assert res["n_matched"] == 1


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def test_assign_split_deterministic_and_valid():
    s = assign_split("a lone astronaut", seed=0)
    assert s in {"train", "val", "test"}
    assert assign_split("a lone astronaut", seed=0) == s  # determinista


def test_assign_split_same_key_no_leakage():
    # La misma clave (prompt) cae siempre en el mismo split → sin fuga.
    m = assign_splits(["p1", "p1", "p2", "p1"], seed=3)
    assert set(m) == {"p1", "p2"}
    assert all(assign_split("p1", seed=3) == m["p1"] for _ in range(5))


def test_assign_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        assign_split("x", ratios=(0.5, 0.3, 0.3))


def test_split_distribution_roughly_respects_ratios():
    keys = [f"prompt-{i}" for i in range(2000)]
    counts = {"train": 0, "val": 0, "test": 0}
    for k in keys:
        counts[assign_split(k, ratios=(0.8, 0.1, 0.1), seed=0)] += 1
    assert counts["train"] / 2000 == pytest.approx(0.8, abs=0.05)


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #


def test_hamming():
    assert hamming(0b0000, 0b0000) == 0
    assert hamming(0b1010, 0b0000) == 2


def test_duplicate_indices_keeps_first():
    # 0 y 2 son idénticos; 2 se marca duplicado, 1 (lejano) no.
    hashes = [0b0000, 0b1111_1111, 0b0000]
    assert duplicate_indices(hashes, threshold=2) == {2}


def test_duplicate_indices_near_threshold():
    # dist(0, 0b11)=2 ≤3 → dup; 0b111100 dista 4 de 0 y 6 de 0b11 → se conserva.
    hashes = [0, 0b11, 0b111100]
    assert duplicate_indices(hashes, threshold=3) == {1}


# --------------------------------------------------------------------------- #
# quality gate + merge
# --------------------------------------------------------------------------- #


def test_passes_gate_all_good():
    ok, reasons = passes_gate(
        {"clip_score": 0.30, "aesthetic": 6.0, "faithfulness": 0.8},
        {"nsfw": 0.01},
    )
    assert ok and reasons == []


def test_passes_gate_collects_reasons():
    ok, reasons = passes_gate(
        {"clip_score": 0.10, "aesthetic": 3.0, "faithfulness": 0.1},
        {"nsfw": 0.9},
        is_duplicate=True,
    )
    assert not ok
    assert set(reasons) == {"duplicate", "low_clip", "low_aesthetic", "low_faithfulness", "nsfw"}


def test_passes_gate_none_scores_do_not_penalize():
    ok, reasons = passes_gate({"faithfulness": None, "clip_score": None}, {"nsfw": None})
    assert ok and reasons == []


def test_merge_eval_writes_dataset_block():
    record = {"id": "abc", "prompt": "p", "dataset": {"license": "L", "code_version": "v1"}}
    out = merge_eval(
        record,
        quality={"clip_score": 0.3, "faithfulness": 0.5},
        safety={"nsfw": 0.02},
        split="train",
        is_duplicate=False,
    )
    ds = out["dataset"]
    assert ds["license"] == "L" and ds["code_version"] == "v1"  # preserva provenance
    assert ds["split"] == "train"
    assert ds["quality"]["passed_gate"] is True
    assert ds["safety"]["is_duplicate"] is False
