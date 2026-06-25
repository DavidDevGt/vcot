import json
import os

import pytest

from vcot.train.distill import build_sft_dataset


def _write_traces(trace, tmp_path, n=2):
    path = os.path.join(tmp_path, "traces.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(n):
            fh.write(json.dumps(trace.model_dump()) + "\n")
    return path


def test_build_messages_dataset(trace, tmp_path):
    traces = _write_traces(trace, tmp_path)
    out = os.path.join(tmp_path, "sft.jsonl")
    n = build_sft_dataset(traces, out, fmt="messages")
    assert n == 2
    with open(out, encoding="utf-8") as fh:
        rows = [json.loads(ln) for ln in fh if ln.strip()]
    assert len(rows) == 2
    assert "messages" in rows[0]


def test_build_tokens_dataset(trace, tmp_path):
    traces = _write_traces(trace, tmp_path)
    out = os.path.join(tmp_path, "tok.jsonl")
    n = build_sft_dataset(traces, out, fmt="tokens")
    assert n == 2
    with open(out, encoding="utf-8") as fh:
        first = json.loads(fh.readline())
    assert set(first) == {"prompt", "completion"}


def test_invalid_format_raises(trace, tmp_path):
    traces = _write_traces(trace, tmp_path)
    with pytest.raises(ValueError):
        build_sft_dataset(traces, os.path.join(tmp_path, "x.jsonl"), fmt="bogus")
