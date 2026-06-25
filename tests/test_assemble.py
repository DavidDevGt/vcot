import json

from vcot.dataset.assemble import assemble_dataset, assemble_record


def _trace(rid="abc", prompt="a lone astronaut"):
    return {
        "id": rid, "prompt": prompt,
        "semantic_plan": {"subject": "a", "environment": "b", "camera": "c",
                          "mood": "d", "dominant_elements": ["e"]},
        "layout": {"entities": [{"id": "a", "kind": "object", "bbox": [0.1, 0.1, 0.9, 0.9]}],
                   "relations": [{"subject": "a", "predicate": "near", "object": "a"}]},
        "composition": {"lens": "35mm", "rule_of_thirds": True, "subject_scale": 0.4,
                        "leading_lines": False, "symmetry": False},
        "lighting": {"key_light": "moon", "fill_light": "low", "rim_light": True, "contrast": "high"},
        "materials": {"materials": {"glass": "wet"}},
        "color_script": {"primary_palette": ["#0F172A"], "temperature": "cold", "saturation": "medium"},
        "meta": {"planner": "Qwen/Qwen3-8B"},
    }


def _render(rid="abc"):
    return {
        "id": rid, "prompt": "enriched render prompt",
        "meta": {"gpu": "A100-80GB", "seed": 42},
        "telemetry": {"render": {"compute_s": 8.0, "rate_usd_per_s": 0.000694, "cost_usd": 0.00555}},
        "final_image": f"/outputs/{rid}_0.webp",
        "final_images": [f"/outputs/{rid}_{i}.webp" for i in range(4)],
        "images": [{"path": f"/outputs/{rid}_{i}.webp", "sha256": f"h{i}", "idx": i,
                    "width": 1024, "height": 1024, "seed": 42} for i in range(4)],
    }


def test_assemble_record_joins_reasoning_and_render():
    rec = assemble_record(_trace(), _render(), license="L", code_version="v1")
    assert rec["enriched_prompt"] == "enriched render prompt"  # el prompt del render
    assert rec["prompt"] == "a lone astronaut"                  # el brief original se preserva
    assert len(rec["images"]) == 4 and rec["images"][0]["sha256"] == "h0"
    assert rec["render"]["projected_cost_usd"] == 0.00555
    assert rec["render"]["projected_gpu"] == "A100-80GB"
    assert rec["meta"]["seed"] == 42
    assert rec["dataset"]["license"] == "L" and rec["dataset"]["code_version"] == "v1"


def test_assemble_record_validates_against_schema():
    from vcot.pipeline.schemas import VCoTTrace

    rec = assemble_record(_trace(), _render(), license="L", code_version="v1")
    VCoTTrace.model_validate(rec)  # no lanza


def test_assemble_dataset_skips_records_without_trace(tmp_path):
    (tmp_path / "abc.trace.json").write_text(json.dumps(_trace("abc")), encoding="utf-8")
    # 'abc' tiene trace; 'zzz' no → se omite.
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps(_render(r)) for r in ["abc", "zzz"]), encoding="utf-8")
    out = tmp_path / "dataset.jsonl"

    n = assemble_dataset(str(records), str(tmp_path), str(out), license="L", code_version="v1")
    assert n == 1
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["abc"]
