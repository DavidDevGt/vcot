import json

from vcot.dataset import SEED_PROMPTS, trace_to_sft, trace_to_token_target


def test_seed_prompts_nonempty_and_unique():
    assert len(SEED_PROMPTS) >= 10
    assert len(set(SEED_PROMPTS)) == len(SEED_PROMPTS)
    assert all(isinstance(p, str) and p.strip() for p in SEED_PROMPTS)


def test_trace_to_sft_shape(trace):
    ex = trace_to_sft(trace)
    msgs = ex["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == trace.prompt
    payload = json.loads(msgs[2]["content"])
    assert set(payload) == {"reasoning", "visual_tokens", "render_prompt"}
    assert set(payload["reasoning"]) == {
        "semantic_plan", "layout", "composition", "lighting", "materials", "color_script"
    }
    assert payload["render_prompt"].endswith("cinematic")


def test_trace_to_token_target(trace):
    tt = trace_to_token_target(trace)
    assert set(tt) == {"prompt", "completion"}
    assert tt["completion"].endswith("RENDER")
