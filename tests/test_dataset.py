import json

import pytest

from vcot.dataset import (
    SEED_PROMPTS,
    generate_prompts,
    trace_to_sft,
    trace_to_token_target,
)


def test_seed_prompts_nonempty_and_unique():
    assert len(SEED_PROMPTS) >= 10
    assert len(set(SEED_PROMPTS)) == len(SEED_PROMPTS)
    assert all(isinstance(p, str) and p.strip() for p in SEED_PROMPTS)


def test_generate_prompts_count_and_uniqueness():
    # Smoke 100 supera el núcleo curado (36) → ejercita el generador.
    prompts = generate_prompts(100)
    assert len(prompts) == 100
    assert len(set(prompts)) == 100  # sin duplicados
    assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_generate_prompts_prepends_curated_core():
    prompts = generate_prompts(100)
    assert prompts[: len(SEED_PROMPTS)] == SEED_PROMPTS


def test_generate_prompts_is_deterministic():
    assert generate_prompts(100, seed=7) == generate_prompts(100, seed=7)
    # Distinta semilla cambia la cola generada (no el núcleo curado).
    a, b = generate_prompts(100, seed=1), generate_prompts(100, seed=2)
    assert a != b
    assert a[: len(SEED_PROMPTS)] == b[: len(SEED_PROMPTS)] == SEED_PROMPTS


def test_generate_prompts_small_n_returns_curated_slice():
    assert generate_prompts(5) == SEED_PROMPTS[:5]
    assert generate_prompts(0) == []


def test_generate_prompts_without_curated_is_unique_and_generated():
    # Sin núcleo curado, el round-robin reparte combinaciones entre los estratos.
    prompts = generate_prompts(36, include_curated=False)
    assert len(prompts) == 36
    assert len(set(prompts)) == 36
    # No es simplemente el núcleo curado en orden: es la expansión generada.
    assert prompts != SEED_PROMPTS


def test_generate_prompts_overflow_raises():
    with pytest.raises(ValueError):
        generate_prompts(10**7)


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
