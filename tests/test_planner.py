import json

import pytest

from vcot.pipeline.llm import FakeLLMClient
from vcot.pipeline.planner import Planner, PlannerError

VALID = {
    "semantic_plan": {
        "subject": "astronaut",
        "environment": "abandoned cathedral",
        "camera": "wide angle",
        "mood": "mysterious",
        "dominant_elements": ["stained glass", "fog"],
    },
    "layout": {
        "canvas": [1024, 1024],
        "entities": [
            {"id": "astronaut", "kind": "character", "bbox": [0.35, 0.45, 0.65, 0.95], "z": 1},
            {"id": "moonlight", "kind": "light", "bbox": [0.0, 0.0, 0.4, 0.3], "z": 3},
        ],
        "relations": [
            {"subject": "astronaut", "predicate": "illuminated_by", "object": "moonlight"}
        ],
    },
    "composition": {
        "lens": "35mm",
        "rule_of_thirds": True,
        "subject_scale": 0.4,
        "leading_lines": True,
        "symmetry": False,
    },
    "lighting": {"key_light": "moonlight", "fill_light": "low", "rim_light": True, "contrast": "high"},
    "materials": {"materials": {"glass": "wet reflective", "stone": "aged gothic"}},
    "color_script": {
        "primary_palette": ["#0F172A", "#3B82F6"],
        "temperature": "cold",
        "saturation": "medium",
    },
}

ORDER = ["semantic_plan", "layout", "composition", "lighting", "materials", "color_script"]


def _all_valid():
    return [VALID[s] for s in ORDER]


def test_plan_happy_path():
    client = FakeLLMClient(_all_valid())
    trace = Planner(client, projected_gpu="A100-40GB").plan("a lone astronaut")

    assert trace.semantic_plan.subject == "astronaut"
    assert len(trace.layout.entities) == 2
    assert set(trace.telemetry) == set(ORDER)
    assert len(client.calls) == 6  # una llamada por etapa, sin reintentos
    assert trace.visual_tokens[-1] == "RENDER"
    assert trace.total_projected_cost_usd >= 0.0
    assert all(t.output_tokens > 0 for t in trace.telemetry.values())
    assert trace.meta["projected_gpu"] == "A100-40GB"


def test_projected_cost_uses_gpu_rate():
    from vcot.telemetry.rates import gpu_rate

    client = FakeLLMClient(_all_valid())
    trace = Planner(client, projected_gpu="H100").plan("x")
    tele = trace.telemetry["semantic_plan"]
    assert tele.rate_usd_per_s == pytest.approx(gpu_rate("H100"))
    assert tele.projected_cost_usd == pytest.approx(tele.compute_s * gpu_rate("H100"))


def test_retry_on_invalid_then_valid():
    bad_comp = dict(VALID["composition"], subject_scale=2.0)  # fuera de [0,1]
    responses = [
        VALID["semantic_plan"],
        VALID["layout"],
        bad_comp,
        VALID["composition"],
        VALID["lighting"],
        VALID["materials"],
        VALID["color_script"],
    ]
    trace = Planner(FakeLLMClient(responses), max_retries=2).plan("x")
    assert trace.telemetry["composition"].retries == 1
    assert trace.composition.subject_scale == 0.4


def test_tolerates_code_fenced_json():
    responses = _all_valid()
    responses[2] = "```json\n" + json.dumps(VALID["composition"]) + "\n```"
    trace = Planner(FakeLLMClient(responses)).plan("x")
    assert trace.telemetry["composition"].retries == 0
    assert trace.composition.lens == "35mm"


def test_tolerates_think_block():
    # Qwen3 y otros modelos de razonamiento pueden anteponer <think>…</think>.
    responses = _all_valid()
    responses[0] = (
        "<think>let me reason about the subject and mood {ignored}</think>\n"
        + json.dumps(VALID["semantic_plan"])
    )
    trace = Planner(FakeLLMClient(responses)).plan("x")
    assert trace.telemetry["semantic_plan"].retries == 0
    assert trace.semantic_plan.subject == "astronaut"


def test_raises_after_exhausting_retries():
    bad_comp = dict(VALID["composition"], subject_scale=2.0)
    responses = [VALID["semantic_plan"], VALID["layout"], bad_comp]
    with pytest.raises(PlannerError):
        Planner(FakeLLMClient(responses), max_retries=0).plan("x")
