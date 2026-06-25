"""Trace → prompt enriquecido para el render N7 (IDEA.md §3.1).

El mecanismo de *conditioning* más simple (y el más barato): serializar la cadena
de decisiones N1–N6 a un prompt rico que FLUX entiende. Es el baseline de §3 — el
punto de partida sobre el que medir mecanismos más fuertes (control espacial,
renders intermedios).

Función pura y determinista: misma traza → mismo prompt.
"""

from __future__ import annotations

from vcot.pipeline.schemas import VCoTTrace


def enrich_prompt(trace: VCoTTrace) -> str:
    """Construye un prompt de render a partir de la traza de razonamiento."""
    p = trace.semantic_plan
    c = trace.composition
    li = trace.lighting
    cs = trace.color_script

    parts = [
        f"{p.subject} in {p.environment}",
        f"{p.camera} shot",
        f"{p.mood} mood",
    ]

    # Composición
    comp = [f"{c.lens} lens", f"subject scale {c.subject_scale:g}"]
    if c.rule_of_thirds:
        comp.append("rule of thirds")
    if c.leading_lines:
        comp.append("leading lines")
    if c.symmetry:
        comp.append("symmetrical composition")
    parts.append("composition: " + ", ".join(comp))

    # Iluminación
    parts.append(
        f"lighting: {li.key_light} key light, {li.fill_light} fill, "
        + ("rim light, " if li.rim_light else "")
        + f"{li.contrast} contrast"
    )

    # Materiales
    mats = ", ".join(f"{surface} {desc}" for surface, desc in trace.materials.materials.items())
    if mats:
        parts.append("materials: " + mats)

    # Color
    palette = ", ".join(cs.primary_palette)
    parts.append(f"color script: {cs.temperature} palette ({palette}), {cs.saturation} saturation")

    # Relaciones espaciales (scene graph → frase legible)
    rels = trace.layout.relations
    if rels:
        readable = {
            "inside": "inside", "on": "on", "behind": "behind",
            "in_front_of": "in front of", "above": "above", "below": "below",
            "left_of": "to the left of", "right_of": "to the right of", "near": "near",
            "casts_shadow_on": "casting a shadow on", "illuminated_by": "illuminated by",
            "passes_through": "passing through", "reflects": "reflecting", "part_of": "part of",
        }
        phrases = [
            f"{r.subject.replace('_', ' ')} {readable.get(r.predicate, r.predicate)} "
            f"{r.object.replace('_', ' ')}"
            for r in rels
        ]
        parts.append("spatial relations: " + "; ".join(phrases))

    # Elementos dominantes
    if p.dominant_elements:
        parts.append("featuring " + ", ".join(p.dominant_elements))

    parts.append("highly detailed, cinematic")
    return ". ".join(parts)
