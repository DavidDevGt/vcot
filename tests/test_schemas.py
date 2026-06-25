import pytest
from pydantic import ValidationError

from vcot.pipeline.schemas import (
    ColorScript,
    Composition,
    Entity,
    Lighting,
    Materials,
    Relation,
    SemanticPlan,
    SpatialLayout,
    StageTelemetry,
    VCoTTrace,
)


def test_semantic_plan_valid():
    p = SemanticPlan(
        subject="astronaut",
        environment="cathedral",
        camera="wide angle",
        mood="mysterious",
        dominant_elements=["fog"],
    )
    assert p.subject == "astronaut"


def test_entity_bbox_out_of_range():
    with pytest.raises(ValidationError):
        Entity(id="x", kind="object", bbox=(0.0, 0.0, 1.2, 0.5))


def test_entity_bbox_degenerate():
    with pytest.raises(ValidationError):
        Entity(id="x", kind="object", bbox=(0.5, 0.5, 0.5, 0.9))  # x1 == x0


def test_entity_full_frame_rejected_for_non_background():
    with pytest.raises(ValidationError):
        Entity(id="moon", kind="light", bbox=(0.0, 0.0, 1.0, 1.0))


def test_entity_full_frame_ok_for_background():
    assert Entity(id="bg", kind="background", bbox=(0.0, 0.0, 1.0, 1.0)).kind == "background"


def test_layout_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        SpatialLayout(
            entities=[
                Entity(id="a", kind="object", bbox=(0.0, 0.0, 0.5, 0.5)),
                Entity(id="a", kind="object", bbox=(0.5, 0.5, 0.9, 0.9)),
            ],
            relations=[Relation(subject="a", predicate="near", object="a")],
        )


def test_layout_relation_must_reference_entities():
    with pytest.raises(ValidationError):
        SpatialLayout(
            entities=[Entity(id="a", kind="object", bbox=(0.0, 0.0, 0.5, 0.5))],
            relations=[Relation(subject="a", predicate="inside", object="ghost")],
        )


def test_layout_relation_allows_implicit_anchor():
    # Una sombra puede caer sobre 'floor' aunque 'floor' no sea entidad.
    layout = SpatialLayout(
        entities=[Entity(id="a", kind="character", bbox=(0.3, 0.3, 0.6, 0.8))],
        relations=[Relation(subject="a", predicate="casts_shadow_on", object="floor")],
    )
    assert layout.relations[0].object == "floor"


def test_relation_predicate_normalizes_synonyms():
    relation = Relation(subject="a", predicate="in", object="b")
    assert relation.predicate == "inside"

    relation = Relation(subject="a", predicate="appear_on", object="b")
    assert relation.predicate == "on"


def test_composition_subject_scale_bounds():
    with pytest.raises(ValidationError):
        Composition(
            lens="35mm",
            rule_of_thirds=True,
            subject_scale=2.0,
            leading_lines=False,
            symmetry=False,
        )


def test_lighting_contrast_literal():
    with pytest.raises(ValidationError):
        Lighting(key_light="moon", fill_light="low", rim_light=True, contrast="extreme")


def test_color_script_rejects_bad_hex():
    with pytest.raises(ValidationError):
        ColorScript(primary_palette=["#zzzzzz"], temperature="cold", saturation="low")


def test_color_script_valid_hex():
    cs = ColorScript(primary_palette=["#0F172A"], temperature="cold", saturation="medium")
    assert cs.primary_palette == ["#0F172A"]


def test_color_script_normalizes_synonyms():
    # Qwen3 dice "cool"/"muted"; deben mapearse a los valores canónicos.
    cs = ColorScript(primary_palette=["#0F172A"], temperature="cool", saturation="muted")
    assert cs.temperature == "cold"
    assert cs.saturation == "low"


def test_materials_rejects_empty_value():
    with pytest.raises(ValidationError):
        Materials(materials={"glass": "  "})


def _minimal_trace(**overrides):
    base = dict(
        id="abc",
        prompt="p",
        semantic_plan=SemanticPlan(
            subject="a", environment="b", camera="c", mood="d", dominant_elements=["e"]
        ),
        layout=SpatialLayout(
            entities=[Entity(id="a", kind="object", bbox=(0.1, 0.1, 0.9, 0.9))],
            relations=[Relation(subject="a", predicate="near", object="a")],
        ),
        composition=Composition(
            lens="35mm", rule_of_thirds=True, subject_scale=0.4,
            leading_lines=False, symmetry=False,
        ),
        lighting=Lighting(key_light="moon", fill_light="low", rim_light=True, contrast="high"),
        materials=Materials(materials={"glass": "wet"}),
        color_script=ColorScript(
            primary_palette=["#0F172A"], temperature="cold", saturation="medium"
        ),
    )
    base.update(overrides)
    return VCoTTrace(**base)


def test_trace_rejects_unknown_telemetry_stage():
    bad = {
        "nope": StageTelemetry(
            compute_s=1.0, rate_usd_per_s=0.0005,
            projected_cost_usd=0.0005, projected_gpu="A100-40GB",
        )
    }
    with pytest.raises(ValidationError):
        _minimal_trace(telemetry=bad)


def test_trace_totals():
    tele = {
        "semantic_plan": StageTelemetry(
            compute_s=1.0, rate_usd_per_s=0.000583,
            projected_cost_usd=0.000583, projected_gpu="A100-40GB",
        ),
        "layout": StageTelemetry(
            compute_s=2.0, rate_usd_per_s=0.000583,
            projected_cost_usd=0.001166, projected_gpu="A100-40GB",
        ),
    }
    trace = _minimal_trace(telemetry=tele)
    assert trace.total_compute_s == pytest.approx(3.0)
    assert trace.total_projected_cost_usd == pytest.approx(0.001749)
