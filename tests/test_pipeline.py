import pytest

from vcot.pipeline.llm import FakeLLMClient
from vcot.pipeline.pipeline import run_pipeline
from vcot.pipeline.planner import Planner


def _fake_render(enriched: str) -> dict:
    # Imita el registro que devuelve modal_app/renderer.py (4 variaciones)
    imgs = [f"/outputs/abc_{i}.webp" for i in range(4)]
    return {
        "final_image": imgs[0],
        "final_images": imgs,
        "telemetry": {
            "render": {
                "compute_s": 2.0, "rate_usd_per_s": 0.000694, "cost_usd": 0.001388,
                "n_variations": 4, "cost_per_image_usd": 0.000347,
            }
        },
        "meta": {"gpu": "A100-80GB"},
    }


def test_run_pipeline_closes_the_loop(valid_responses):
    planner = Planner(FakeLLMClient(valid_responses), projected_gpu="A100-40GB")
    trace = run_pipeline("a lone astronaut", planner, _fake_render)

    assert trace.enriched_prompt and "astronaut" in trace.enriched_prompt
    assert trace.final_image == "/outputs/abc_0.webp"
    assert trace.final_images == [f"/outputs/abc_{i}.webp" for i in range(4)]
    assert trace.render is not None
    assert trace.render.projected_gpu == "A100-80GB"
    assert trace.render.projected_cost_usd == pytest.approx(0.001388)


def test_e2e_cost_adds_reasoning_and_render(valid_responses):
    planner = Planner(FakeLLMClient(valid_responses), projected_gpu="A100-40GB")
    trace = run_pipeline("x", planner, _fake_render)
    assert trace.e2e_cost_usd == pytest.approx(
        trace.total_projected_cost_usd + 0.001388
    )
    assert trace.e2e_compute_s == pytest.approx(trace.total_compute_s + 2.0)
