"""Prompts del planner V-CoT (IDEA.md §2, §4.1).

El planner genera la cadena **etapa por etapa**: cada llamada al LLM ve el prompt
original y *todas las decisiones anteriores*, y produce solo la siguiente. Eso es
lo que hace la cadena un verdadero chain-of-thought visual (y no un único volcado
de JSON): cada etapa está condicionada por la previa y es criticable por separado
(§5.3).

Los prompts van en inglés a propósito: mejor rendimiento de los LLM y consistencia
con los ejemplos JSON de IDEA.md. Los comentarios del código siguen en español.
"""

from __future__ import annotations

import json
from typing import Dict

from vcot.pipeline.schemas import STAGE_LABELS

SYSTEM_PROMPT = (
    "You are an expert visual director (cinematographer + concept artist) who "
    "thinks in an explicit, step-by-step pipeline before any pixel exists: "
    "semantic plan, spatial layout, composition, lighting, materials, color. "
    "Reason about THIS specific scene and make deliberate, scene-specific "
    "decisions — never fall back on defaults or templates. Two different prompts "
    "must yield clearly different compositions, layouts and palettes; if you find "
    "yourself reusing the same values (e.g. always 35mm, always rule_of_thirds, "
    "always subject_scale 0.6), you are NOT reasoning. "
    "Always respond in ENGLISH, regardless of the input language. "
    "You ALWAYS answer with a single valid JSON object that matches the requested "
    "schema exactly — no prose, no markdown, no code fences, no comments."
)

#: Instrucción por etapa: qué decisión tomar en este paso de la cadena.
_STAGE_INSTRUCTIONS: Dict[str, str] = {
    "semantic_plan": (
        "STEP N1 — SEMANTIC SCENE PLAN. Before drawing anything, decide the "
        "semantics of the scene: the subject, the environment, the camera "
        "framing, the overall mood, and the dominant visual elements. No "
        "geometry, no pixels — pure intent."
    ),
    "layout": (
        "STEP N2 — SPATIAL LAYOUT (scene graph). ALWAYS include the main "
        "environment/setting as a `background` entity so other things can be placed "
        "relative to it. For each entity set `kind` (object | character | light | "
        "atmosphere | shadow | background), a bounding box [x0,y0,x1,y1] in 0..1 "
        "(origin top-left) and depth `z` (higher = further back). Only physical "
        "objects/characters get tight boxes; lights, atmosphere and shadows must be "
        "placed WHERE THEY ACTUALLY APPEAR (never the full frame) — only "
        "`background` may span the whole canvas. Then add `relations`: edges "
        "{subject, predicate, object} between entity ids, e.g. astronaut inside "
        "cathedral, moonlight passes_through stained_glass, astronaut "
        "casts_shadow_on floor (a shadow falls on a surface like a floor, not on "
        "another shadow). Relations may reference implicit surfaces — floor, "
        "ground, wall, ceiling, sky, water — WITHOUT listing them as entities; any "
        "other id used in a relation MUST be a declared entity. Give each entity a "
        "TIGHT box sized to how much of the frame it really occupies — do NOT "
        "default everything to ~[0.1,0.1,0.9,0.9]; boxes should differ. Be concise: "
        "4–8 entities and only the relations that genuinely matter (2–5 typical) — "
        "never invent one to fill a quota. Use lowercase snake_case ids."
    ),
    "composition": (
        "STEP N3 — COMPOSITION. Make cinematographic choices DERIVED from this "
        "scene — do not default. Pick a lens that fits (wide for landscapes/scale, "
        "longer for intimate portraits). `subject_scale` MUST match the framing: "
        "close-up 0.6–0.9, medium 0.3–0.6, wide/landscape 0.05–0.3. Use "
        "rule_of_thirds, leading_lines and symmetry ONLY when they genuinely suit "
        "this scene (a centered symmetric subject is not rule-of-thirds). "
        "Different scenes must produce different values."
    ),
    "lighting": (
        "STEP N4 — LIGHTING DESIGN. Design light that fits THIS scene's mood and "
        "time: the key light source, the fill light level, whether there is a rim "
        "light, and the overall contrast. A bright noon scene and a candlelit one "
        "must differ."
    ),
    "materials": (
        "STEP N5 — MATERIAL DEFINITION. Map the key physical surfaces (the objects "
        "from the layout, not lights/atmosphere) to a short, specific material "
        "description (e.g. 'oak_bark': 'cracked weathered wood')."
    ),
    "color_script": (
        "STEP N6 — COLOR SCRIPT. Choose a palette that expresses THIS scene's mood "
        "— not a generic grey set. Give a primary palette of hex colors (#RRGGBB), "
        "a temperature and a saturation level consistent with the lighting."
    ),
}


def build_stage_prompt(
    stage: str,
    user_prompt: str,
    prior: Dict[str, dict],
    schema: dict,
) -> str:
    """Construye el prompt de usuario para una etapa de la cadena.

    Parameters
    ----------
    stage:
        Clave de etapa (``"semantic_plan"`` … ``"color_script"``).
    user_prompt:
        El prompt original del que parte toda la cadena.
    prior:
        Decisiones ya tomadas ``{stage: dict}`` en orden, para condicionar.
    schema:
        JSON Schema (de pydantic) que la salida debe cumplir.
    """
    parts = [
        f"ORIGINAL PROMPT:\n{user_prompt}",
    ]
    if prior:
        # Resumen compacto (no JSON completo): recorta el contexto que arrastra
        # cada etapa — sobre todo el layout, que inflaba el input de N4–N6.
        rendered = "\n".join(
            f"- {STAGE_LABELS[s]} {s}: {_compact_prior(s, v)}"
            for s, v in prior.items()
        )
        parts.append("DECISIONS SO FAR:\n" + rendered)
    parts.append(_STAGE_INSTRUCTIONS[stage])
    parts.append(
        "Respond with a single JSON object that validates against this JSON "
        "Schema:\n" + json.dumps(schema, ensure_ascii=False)
    )
    return "\n\n".join(parts)


def repair_prompt(error: str) -> str:
    """Mensaje de reintento cuando la salida no valida (bucle de §5.3)."""
    return (
        "Your previous answer was not valid for the schema. Error:\n"
        f"{error}\n"
        "Return ONLY a corrected JSON object. No prose, no code fences."
    )


def _compact_prior(stage: str, d: dict) -> str:
    """Resumen de una etapa previa para el contexto (mucho más corto que el JSON).

    El layout omite bboxes/z (lo que más tokens costaba): las etapas posteriores
    razonan con qué hay y cómo se relaciona, no con coordenadas exactas.
    """
    try:
        if stage == "semantic_plan":
            return (
                f"subject={d['subject']}; environment={d['environment']}; "
                f"camera={d['camera']}; mood={d['mood']}; "
                f"elements={', '.join(d.get('dominant_elements', []))}"
            )
        if stage == "layout":
            ents = ", ".join(f"{e['id']}({e['kind']})" for e in d.get("entities", []))
            rels = ", ".join(
                f"{r['subject']} {r['predicate']} {r['object']}" for r in d.get("relations", [])
            )
            return f"entities=[{ents}]; relations=[{rels}]"
        if stage == "composition":
            return (
                f"lens={d['lens']}; subject_scale={d['subject_scale']}; "
                f"rule_of_thirds={d['rule_of_thirds']}; leading_lines={d['leading_lines']}; "
                f"symmetry={d['symmetry']}"
            )
        if stage == "lighting":
            return (
                f"key_light={d['key_light']}; fill={d['fill_light']}; "
                f"rim={d['rim_light']}; contrast={d['contrast']}"
            )
        if stage == "materials":
            return "; ".join(f"{k}={v}" for k, v in d.get("materials", {}).items())
        if stage == "color_script":
            return (
                f"palette={', '.join(d.get('primary_palette', []))}; "
                f"temperature={d['temperature']}; saturation={d['saturation']}"
            )
    except (KeyError, TypeError):
        pass
    return json.dumps(d, ensure_ascii=False)
