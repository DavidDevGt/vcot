import pytest

from vcot.pipeline.llm import FakeLLMClient
from vcot.pipeline.planner import Planner

# Salidas válidas por etapa para el FakeLLMClient (cadena completa N1–N6).
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


@pytest.fixture
def valid_responses():
    return [VALID[s] for s in ORDER]


@pytest.fixture
def trace(valid_responses):
    """Una VCoTTrace válida (solo razonamiento N1–N6) construida con el fake client."""
    return Planner(FakeLLMClient(valid_responses), projected_gpu="A100-40GB").plan(
        "a lone astronaut in a gothic cathedral"
    )
