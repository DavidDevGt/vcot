import os

import pytest

from vcot.reporting.runlog import append_run, load_runs, track_run


def test_append_and_load_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "runs.jsonl")
    append_run(path, {"kind": "planner", "n_items": 3})
    append_run(path, {"kind": "dataset", "n_items": 10})
    runs = load_runs(path)
    assert [r["kind"] for r in runs] == ["planner", "dataset"]
    assert runs[1]["n_items"] == 10


def test_load_missing_returns_empty(tmp_path):
    assert load_runs(os.path.join(tmp_path, "nope.jsonl")) == []


def test_track_run_ok(tmp_path):
    path = os.path.join(tmp_path, "runs.jsonl")
    with track_run(path, kind="dataset", model="Qwen/Qwen3-8B", gpu="A100-40GB") as run:
        run["n_items"] = 5
        run["total_cost_usd"] = 0.12
    (rec,) = load_runs(path)
    assert rec["status"] == "ok"
    assert rec["kind"] == "dataset"
    assert rec["model"] == "Qwen/Qwen3-8B"
    assert rec["n_items"] == 5
    assert rec["duration_s"] >= 0.0
    assert rec["id"] and rec["started_at"]


def test_track_run_records_errors_and_reraises(tmp_path):
    path = os.path.join(tmp_path, "runs.jsonl")
    with pytest.raises(RuntimeError):
        with track_run(path, kind="pipeline"):
            raise RuntimeError("boom")
    (rec,) = load_runs(path)
    assert rec["status"] == "error"
    assert "boom" in rec["error"]
