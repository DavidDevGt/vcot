import os

from vcot.analysis.aggregate import (
    _pct,
    format_report,
    load_jsonl,
    summarize,
    write_csv,
)


def test_pct_interpolates():
    assert _pct([10, 20, 30], 50) == 20
    assert _pct([10, 20], 50) == 15
    assert _pct([], 90) == 0.0
    assert _pct([5], 90) == 5


def _records(trace):
    r1 = trace.model_dump()  # solo razonamiento
    r2 = trace.model_dump()
    r2["render"] = {
        "compute_s": 2.0,
        "rate_usd_per_s": 0.000694,
        "projected_cost_usd": 0.001388,
        "projected_gpu": "A100-80GB",
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_per_s": 0.0,
        "retries": 0,
    }
    return [r1, r2]


def test_summarize_counts_and_render(trace):
    s = summarize(_records(trace))
    assert s["n"] == 2
    assert set(s["per_stage"]) == {
        "semantic_plan", "layout", "composition", "lighting", "materials", "color_script"
    }
    assert s["render"]["n"] == 1
    assert s["e2e"]["projected_1k"] == s["e2e"]["cost_mean"] * 1000
    assert s["e2e"]["cost_total"] > 0


def test_format_report_and_csv(trace, tmp_path):
    s = summarize(_records(trace))
    report = format_report(s)
    assert "V-CoT" in report and "E2E" in report

    csv_path = os.path.join(tmp_path, "cost.csv")
    write_csv(s, csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path, encoding="utf-8") as fh:
        head = fh.readline()
    assert "stage" in head and "cost_mean" in head


def test_load_jsonl_skips_blank_lines(trace, tmp_path):
    path = os.path.join(tmp_path, "traces.jsonl")
    import json

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(trace.model_dump()) + "\n\n")
    assert len(load_jsonl(path)) == 1
