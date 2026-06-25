import json
import tarfile

from vcot.dataset.pack import build_index_row, pack, render_datacard, summarize


def _record(rid, split, passed, prompt="a lone astronaut"):
    return {
        "id": rid,
        "prompt": prompt,
        "meta": {"planner": "Qwen/Qwen3-8B", "n_variations": 2},
        "layout": {"entities": []},
        "images": [
            {"path": f"/o/{rid}_0.webp", "sha256": "h0", "idx": 0, "width": 8, "height": 8},
            {"path": f"/o/{rid}_1.webp", "sha256": "h1", "idx": 1, "width": 8, "height": 8},
        ],
        "dataset": {
            "license": "FLUX.2 (non-commercial) + Qwen3 (Apache-2.0)",
            "code_version": "abc123",
            "split": split,
            "quality": {
                "clip_score": 0.30, "aesthetic": 6.0, "faithfulness": 0.5,
                "passed_gate": passed, "gate_reasons": [] if passed else ["low_clip"],
            },
            "safety": {"nsfw": 0.01, "is_duplicate": False},
        },
    }


def _write_dataset(tmp_path, records):
    images = tmp_path / "outputs"
    images.mkdir()
    for r in records:
        for i in range(2):
            (images / f"{r['id']}_{i}.webp").write_bytes(b"RIFF....WEBPfake")
    traces = tmp_path / "traces.eval.jsonl"
    traces.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(traces), str(images)


def test_pack_writes_shards_index_manifest_datacard(tmp_path):
    records = [
        _record("aaa", "train", True),
        _record("bbb", "val", True),
        _record("ccc", "test", False),
    ]
    traces, images = _write_dataset(tmp_path, records)
    out = tmp_path / "dataset"

    manifest = pack(traces, images, str(out), shard_size=256)

    assert manifest["n_samples"] == 3
    assert manifest["n_images"] == 6
    assert manifest["models"]["planner"] == "Qwen/Qwen3-8B"
    assert manifest["gate"] == {"passed": 2, "failed": 1, "unknown": 0}

    # Artefactos en disco.
    assert (out / "manifest.json").exists()
    assert (out / "index.jsonl").exists()
    assert (out / "DATACARD.md").exists()
    shard = out / "shards" / "shard-00000.tar"
    assert shard.exists()

    # El shard contiene el sidecar JSON + las 2 imágenes por muestra.
    with tarfile.open(shard) as tar:
        names = set(tar.getnames())
    assert "aaa.json" in names
    assert {"aaa.0.webp", "aaa.1.webp"} <= names

    # index.jsonl: una fila por muestra con split + gate.
    rows = [json.loads(ln) for ln in (out / "index.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == {"aaa", "bbb", "ccc"}
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_pack_shard_size_splits_into_multiple_tars(tmp_path):
    records = [_record(f"id{i}", "train", True) for i in range(5)]
    traces, images = _write_dataset(tmp_path, records)
    out = tmp_path / "dataset"

    manifest = pack(traces, images, str(out), shard_size=2)
    assert len(manifest["shards"]) == 3  # 5 muestras / 2 → 3 shards
    assert (out / "shards" / "shard-00002.tar").exists()


def test_pack_only_passed_filters(tmp_path):
    records = [_record("ok", "train", True), _record("bad", "train", False)]
    traces, images = _write_dataset(tmp_path, records)
    out = tmp_path / "dataset"

    manifest = pack(traces, images, str(out), only_passed=True)
    assert manifest["n_samples"] == 1
    rows = [json.loads(ln) for ln in (out / "index.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["ok"]


def test_summarize_counts_and_means():
    records = [_record("a", "train", True), _record("b", "test", False)]
    stats = summarize(records)
    assert stats["gate"] == {"passed": 1, "failed": 1, "unknown": 0}
    assert stats["means"]["clip_score"] == 0.30
    assert stats["splits"] == {"train": 1, "val": 0, "test": 1}


def test_render_datacard_has_sections():
    records = [_record("a", "train", True)]
    traces = render_datacard(
        {
            "name": "vcot-dataset", "version": "0.1.0", "schema": "vcot-dataset/1.0",
            "created_at": "2026-06-25T00:00:00+00:00", "code_version": "abc",
            "license": "L", "n_samples": 1, "n_images": 2,
            "splits": {"train": 1}, "gate": {"passed": 1, "failed": 0, "unknown": 0},
            "metric_means": {"clip_score": 0.3, "aesthetic": 6.0, "faithfulness": 0.5},
            "n_duplicates": 0,
            "models": {"planner": "Qwen/Qwen3-8B", "renderer": "FLUX.2-klein-9B"},
        }
    )
    for section in ["# Datacard", "## Motivation", "## Composition", "## Uses", "## Distribution"]:
        assert section in traces
