from vcot.pipeline.schemas import (
    ColorScript,
    Composition,
    Entity,
    Lighting,
    Materials,
    Relation,
    SemanticPlan,
    SpatialLayout,
    VCoTTrace,
)
from vcot.pipeline.visual_tokens import _cell, to_visual_tokens


def _trace():
    return VCoTTrace(
        id="abc",
        prompt="p",
        semantic_plan=SemanticPlan(
            subject="astronaut",
            environment="abandoned cathedral",
            camera="wide angle",
            mood="mysterious",
            dominant_elements=["stained glass", "fog"],
        ),
        # centro en (0.5, 0.5) → 'center'
        layout=SpatialLayout(
            entities=[
                Entity(id="astronaut", kind="character", bbox=(0.35, 0.45, 0.65, 0.55)),
                Entity(id="cathedral", kind="background", bbox=(0.0, 0.0, 1.0, 1.0)),
            ],
            relations=[Relation(subject="astronaut", predicate="inside", object="cathedral")],
        ),
        composition=Composition(
            lens="35mm", rule_of_thirds=True, subject_scale=0.4,
            leading_lines=True, symmetry=False,
        ),
        lighting=Lighting(key_light="moonlight", fill_light="low", rim_light=True, contrast="high"),
        materials=Materials(materials={"glass": "wet reflective"}),
        color_script=ColorScript(
            primary_palette=["#0F172A", "#3B82F6"], temperature="cold", saturation="medium"
        ),
    )


def test_tokens_cover_all_stages_and_end_with_render():
    tokens = to_visual_tokens(_trace())
    assert tokens[-1] == "RENDER"
    for expected in [
        "PLAN_SUBJ:astronaut",
        "ENV:abandoned-cathedral",
        "ELEM:stained-glass",
        "AT:astronaut:character:center",
        "REL:astronaut:inside:cathedral",
        "LENS_35MM",
        "THIRDS",
        "LEADING",
        "LIGHT_KEY:moonlight",
        "FILL_LOW",
        "RIM",
        "CONTRAST_HIGH",
        "MAT:glass:wet-reflective",
        "COLOR_COLD",
        "SAT_MEDIUM",
        "PAL:#0f172a",
    ]:
        assert expected in tokens, expected


def test_symmetry_absent_when_false():
    assert "SYMMETRY" not in to_visual_tokens(_trace())


def test_cell_mapping():
    assert _cell((0.0, 0.0, 0.2, 0.2)) == "top-left"
    assert _cell((0.8, 0.8, 1.0, 1.0)) == "bottom-right"
    assert _cell((0.4, 0.4, 0.6, 0.6)) == "center"
    assert _cell((0.4, 0.0, 0.6, 0.2)) == "top"
    assert _cell((0.0, 0.4, 0.2, 0.6)) == "left"
