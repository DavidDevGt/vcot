"""Serialización traza → Visual Tokens (IDEA.md §2.2).

Comprime una :class:`~vcot.pipeline.schemas.VCoTTrace` a una secuencia compacta y
discreta de tokens de razonamiento visual::

    PLAN_SUBJ:astronaut  ENV:cathedral  AT:astronaut:center  LENS_35MM
    LIGHT_KEY:moonlight  CONTRAST_HIGH  COLOR_COLD  SAT_MEDIUM  RENDER

Esto convierte la imagen en una *secuencia de razonamiento* y habilita las
arquitecturas autoregresivas / MoE de §5.2. La transformación es determinista.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from vcot.pipeline.schemas import VCoTTrace

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def _cell(bbox: Tuple[float, float, float, float]) -> str:
    """Celda 3×3 del centro de la bbox: 'center', 'top-left', 'bottom-right'…"""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "mid")
    if row == "mid" and col == "center":
        return "center"
    if row == "mid":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def to_visual_tokens(trace: VCoTTrace) -> List[str]:
    """Convierte la traza completa en su secuencia de Visual Tokens."""
    t: List[str] = []

    # N1 — Semantic plan
    p = trace.semantic_plan
    t += [
        f"PLAN_SUBJ:{_slug(p.subject)}",
        f"ENV:{_slug(p.environment)}",
        f"CAM:{_slug(p.camera)}",
        f"MOOD:{_slug(p.mood)}",
    ]
    t += [f"ELEM:{_slug(e)}" for e in p.dominant_elements]

    # N2 — Layout (posición + tipo + relaciones del scene graph)
    for ent in trace.layout.entities:
        t.append(f"AT:{_slug(ent.id)}:{ent.kind}:{_cell(ent.bbox)}")
    for rel in trace.layout.relations:
        t.append(f"REL:{_slug(rel.subject)}:{rel.predicate}:{_slug(rel.object)}")

    # N3 — Composition
    c = trace.composition
    t.append("LENS_" + _slug(c.lens).upper().replace("-", ""))
    if c.rule_of_thirds:
        t.append("THIRDS")
    t.append(f"SCALE:{c.subject_scale:g}")
    if c.leading_lines:
        t.append("LEADING")
    if c.symmetry:
        t.append("SYMMETRY")

    # N4 — Lighting
    li = trace.lighting
    t.append(f"LIGHT_KEY:{_slug(li.key_light)}")
    t.append(f"FILL_{li.fill_light.upper()}")
    if li.rim_light:
        t.append("RIM")
    t.append(f"CONTRAST_{li.contrast.upper()}")

    # N5 — Materials
    for surface, desc in trace.materials.materials.items():
        t.append(f"MAT:{_slug(surface)}:{_slug(desc)}")

    # N6 — Color script
    cs = trace.color_script
    t.append(f"COLOR_{cs.temperature.upper()}")
    t.append(f"SAT_{cs.saturation.upper()}")
    t += [f"PAL:{hex_color.lower()}" for hex_color in cs.primary_palette]

    t.append("RENDER")
    return t
