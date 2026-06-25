from vcot.pipeline.enrich import enrich_prompt


def test_enrich_includes_key_decisions(trace):
    prompt = enrich_prompt(trace)
    for expected in [
        "astronaut",
        "abandoned cathedral",
        "wide angle",
        "35mm lens",
        "rule of thirds",
        "moonlight key light",
        "high contrast",
        "glass wet reflective",
        "cold palette",
        "#0F172A",
        "medium saturation",
        "astronaut illuminated by moonlight",  # relación del scene graph (§3.1)
    ]:
        assert expected in prompt, expected
    assert prompt.endswith("cinematic")


def test_enrich_is_deterministic(trace):
    assert enrich_prompt(trace) == enrich_prompt(trace)
