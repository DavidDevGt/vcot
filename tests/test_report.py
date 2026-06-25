import json
import os

from vcot.reporting.report import build_report, render_markdown, write_bundle


def _with_render(rec: dict) -> dict:
    rec = dict(rec)
    rec["final_image"] = "/outputs/x.webp"
    rec["render"] = {
        "compute_s": 2.0,
        "rate_usd_per_s": 0.000694,
        "projected_cost_usd": 0.001388,
        "projected_gpu": "A100-80GB",
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens_per_s": 0.0,
        "retries": 0,
    }
    return rec


def _records(trace):
    return [trace.model_dump(), _with_render(trace.model_dump())]


def test_build_report_structure(trace):
    rep = build_report(_records(trace))
    assert rep["n_traces"] == 2
    assert rep["n_with_image"] == 1
    assert set(rep["cost"]) == {"n", "per_stage", "reasoning", "render", "e2e"}
    assert rep["total_dataset_cost_usd"] >= 0.0
    assert set(rep["latency"]) >= {"e2e_p50_s", "e2e_p90_s", "e2e_p99_s"}
    assert rep["quality"]["first_try_valid_rate"] == 1.0
    assert rep["date_range"] is not None  # meta.created_at presente
    assert "A100-40GB" in rep["rates_usd_per_s"]


def test_build_report_empty():
    rep = build_report([])
    assert rep["n_traces"] == 0


def test_build_report_with_ledger(trace):
    runs = [
        {"kind": "dataset", "total_cost_usd": 0.5, "n_items": 2},
        {"kind": "pipeline", "total_cost_usd": 0.01, "n_items": 1},
    ]
    rep = build_report(_records(trace), runs)
    assert rep["ledger"]["n_runs"] == 2
    assert rep["ledger"]["by_kind"] == {"dataset": 1, "pipeline": 1}


def test_render_markdown_sections(trace):
    md = render_markdown(build_report(_records(trace)))
    for header in [
        "# V-CoT — Informe de inferencia y coste",
        "## 1. Resumen ejecutivo",
        "## 2. Coste y latencia por etapa",
        "## 3. Render (N7)",
        "## 6. Registro de ejecuciones",
    ]:
        assert header in md, header


def test_write_bundle_creates_files(trace, tmp_path):
    runs = [{"kind": "dataset", "total_cost_usd": 0.5, "n_items": 2, "started_at": "x", "status": "ok"}]
    rep = build_report(_records(trace), runs)
    paths = write_bundle(rep, str(tmp_path), _records(trace), runs)

    assert os.path.exists(paths["markdown"])
    assert os.path.exists(paths["json"])
    assert os.path.exists(paths["cost_csv"])
    assert os.path.exists(paths["runs_csv"])
    assert os.path.exists(paths["latest"])
    # report.json es JSON válido
    with open(paths["json"], encoding="utf-8") as fh:
        assert json.load(fh)["n_traces"] == 2
