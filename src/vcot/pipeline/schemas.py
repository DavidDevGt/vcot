"""Esquemas de las etapas de razonamiento V-CoT (IDEA.md §2.1).

Cada etapa N1–N6 es una **decisión observable y validable** antes de que exista
un solo píxel. Modelarlas con pydantic nos da tres cosas que IDEA.md pide
explícitamente: salida estructurada del LLM, validación (la propiedad
"criticable" de §5.3) y serialización del registro por muestra (§4.2).

El orden canónico de la cadena es::

    Prompt → N1 plan → N2 layout → N3 composition → N4 lighting → N5 materials → N6 color

`N7` (render) vive aparte en ``modal_app/renderer.py``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Vocabularios controlados (habilitan la crítica/validación de §5.3)
# --------------------------------------------------------------------------- #

Contrast = Literal["low", "medium", "high"]
Level = Literal["low", "medium", "high"]
Temperature = Literal["cold", "neutral", "warm"]

#: Tipo de entidad — separa objeto físico (con bbox real) de efecto/atmósfera.
EntityKind = Literal["object", "character", "light", "atmosphere", "shadow", "background"]

#: Predicados del scene graph (aristas entre entidades).
Predicate = Literal[
    "inside", "on", "behind", "in_front_of", "above", "below",
    "left_of", "right_of", "near",
    "casts_shadow_on", "illuminated_by", "passes_through", "reflects", "part_of",
]

#: Sinónimos comunes que los modelos pueden emitir y que deben normalizarse a
#: vocabulario canónico para que la etapa pueda validar con robustez.
PREDICATE_SYNONYMS: dict[str, str] = {
    "in": "inside",
    "appear_on": "on",
    "appear_above": "above",
    "appear_below": "below",
    "appear_behind": "behind",
    "appear_in_front_of": "in_front_of",
    "appear_in_front": "in_front_of",
    "in_front": "in_front_of",
    "in_frontof": "in_front_of",
    "on_top_of": "on",
    "onto": "on",
    "inside_of": "inside",
    "partof": "part_of",
    "part_of": "part_of",
}

#: Superficies/anclas implícitas válidas como extremo de una relación aunque no
#: se declaren como entidad (p.ej. una sombra cae sobre el `floor`). Evita que el
#: modelo tenga que inventar entidades — y que el validador degrade la relación.
IMPLICIT_ANCHORS = frozenset(
    {"floor", "ground", "ceiling", "wall", "walls", "sky", "horizon", "water",
     "background", "foreground", "offscreen"}
)

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{6})$")


# --------------------------------------------------------------------------- #
# N1 — Semantic Scene Plan (semántica pura, sin píxeles)
# --------------------------------------------------------------------------- #


class SemanticPlan(BaseModel):
    """Storyboard mental: qué hay en la escena, sin geometría todavía."""

    subject: str = Field(..., min_length=1)
    environment: str = Field(..., min_length=1)
    camera: str = Field(..., min_length=1)
    mood: str = Field(..., min_length=1)
    dominant_elements: List[str] = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# N2 — Spatial Layout (scene graph + geometría aproximada)
# --------------------------------------------------------------------------- #


class Entity(BaseModel):
    """Una entidad de la escena.

    ``kind`` distingue un objeto físico (que ocupa una bbox real) de un efecto
    visual (luz, atmósfera, sombra) — evita que la luna o una sombra se modelen
    como objetos de pantalla completa. ``bbox`` es ``[x0, y0, x1, y1]`` normalizado
    (0–1), origen arriba-izquierda. ``z`` = profundidad (mayor = más al fondo).
    """

    id: str = Field(..., min_length=1)
    kind: EntityKind
    bbox: Tuple[float, float, float, float]
    z: int = 0

    @field_validator("bbox")
    @classmethod
    def _bbox_in_unit_square(
        cls, v: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = v
        for coord in v:
            if not 0.0 <= coord <= 1.0:
                raise ValueError(f"bbox fuera de [0,1]: {v}")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"bbox degenerada (x1<=x0 o y1<=y0): {v}")
        return v

    @model_validator(mode="after")
    def _no_full_frame_effects(self) -> "Entity":
        x0, y0, x1, y1 = self.bbox
        if (x1 - x0) * (y1 - y0) >= 0.98 and self.kind != "background":
            raise ValueError(
                f"entidad '{self.id}' ({self.kind}) ocupa casi todo el lienzo; "
                "solo 'background' puede. Coloca el efecto donde realmente aparece."
            )
        return self


class Relation(BaseModel):
    """Arista del scene graph: ``subject —predicate→ object`` (ids de entidad)."""

    subject: str = Field(..., min_length=1)

    @field_validator("predicate", mode="before")
    @classmethod
    def _normalize_predicate(cls, v):
        if isinstance(v, str):
            norm = v.strip().lower()
            return PREDICATE_SYNONYMS.get(norm, norm)
        return v

    predicate: Predicate
    object: str = Field(..., min_length=1)


class SpatialLayout(BaseModel):
    """Scene graph: entidades posicionadas + **relaciones** entre ellas."""

    canvas: Tuple[int, int] = (1024, 1024)
    entities: List[Entity] = Field(..., min_length=1)
    relations: List[Relation] = Field(..., min_length=1)

    @field_validator("entities")
    @classmethod
    def _unique_ids(cls, v: List[Entity]) -> List[Entity]:
        ids = [e.id for e in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ids de entidad duplicados: {ids}")
        return v

    @model_validator(mode="after")
    def _relations_reference_entities(self) -> "SpatialLayout":
        valid = {e.id for e in self.entities} | IMPLICIT_ANCHORS
        for r in self.relations:
            if r.subject not in valid or r.object not in valid:
                raise ValueError(
                    f"relación con id inexistente: {r.subject} {r.predicate} {r.object}; "
                    f"ids válidos: {sorted(valid)}"
                )
        return self


# --------------------------------------------------------------------------- #
# N3 — Composition (decisiones cinematográficas)
# --------------------------------------------------------------------------- #


class Composition(BaseModel):
    lens: str = Field(..., min_length=1)  # p.ej. "35mm"
    rule_of_thirds: bool
    subject_scale: float = Field(..., ge=0.0, le=1.0)
    leading_lines: bool
    symmetry: bool


# --------------------------------------------------------------------------- #
# N4 — Lighting Design (esquema de luz)
# --------------------------------------------------------------------------- #


class Lighting(BaseModel):
    key_light: str = Field(..., min_length=1)
    fill_light: Level
    rim_light: bool
    contrast: Contrast


# --------------------------------------------------------------------------- #
# N5 — Material Definition (superficies)
# --------------------------------------------------------------------------- #


class Materials(BaseModel):
    """Mapa superficie → descripción (p.ej. ``{"glass": "wet reflective"}``)."""

    materials: Dict[str, str] = Field(..., min_length=1)

    @field_validator("materials")
    @classmethod
    def _non_empty_values(cls, v: Dict[str, str]) -> Dict[str, str]:
        for k, val in v.items():
            if not k.strip() or not val.strip():
                raise ValueError(f"material con clave/valor vacío: {k!r}={val!r}")
        return v


# --------------------------------------------------------------------------- #
# N6 — Color Script (paleta y temperatura)
# --------------------------------------------------------------------------- #


class ColorScript(BaseModel):
    primary_palette: List[str] = Field(..., min_length=1)
    temperature: Temperature
    saturation: Level

    @field_validator("temperature", mode="before")
    @classmethod
    def _norm_temperature(cls, v):
        # Qwen3 suele decir "cool" en vez de "cold" (causaba retry sistemático en N6).
        if isinstance(v, str):
            v = v.strip().lower()
            return {"cool": "cold", "chilly": "cold", "icy": "cold", "hot": "warm"}.get(v, v)
        return v

    @field_validator("saturation", mode="before")
    @classmethod
    def _norm_saturation(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            return {
                "muted": "low", "desaturated": "low", "dull": "low",
                "moderate": "medium", "mid": "medium",
                "vivid": "high", "vibrant": "high", "saturated": "high",
            }.get(v, v)
        return v

    @field_validator("primary_palette")
    @classmethod
    def _valid_hex(cls, v: List[str]) -> List[str]:
        for color in v:
            if not _HEX_RE.match(color):
                raise ValueError(f"color no es hex #RRGGBB: {color!r}")
        return v


# --------------------------------------------------------------------------- #
# Registro de la cadena completa
# --------------------------------------------------------------------------- #

#: Orden canónico de las etapas: clave de etapa → modelo pydantic.
STAGE_MODELS: Dict[str, type[BaseModel]] = {
    "semantic_plan": SemanticPlan,
    "layout": SpatialLayout,
    "composition": Composition,
    "lighting": Lighting,
    "materials": Materials,
    "color_script": ColorScript,
}

#: Etiqueta Nk legible por etapa (para logs/telemetría).
STAGE_LABELS: Dict[str, str] = {
    "semantic_plan": "N1",
    "layout": "N2",
    "composition": "N3",
    "lighting": "N4",
    "materials": "N5",
    "color_script": "N6",
}


class StageTelemetry(BaseModel):
    """Métricas de inferencia de una etapa (IDEA.md §6)."""

    compute_s: float
    rate_usd_per_s: float
    projected_cost_usd: float
    projected_gpu: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_s: float = 0.0
    retries: int = 0
    last_error: Optional[str] = None  # por qué falló el 1er intento (diagnóstico)


class VCoTTrace(BaseModel):
    """La traza completa de pensamiento visual de un prompt (IDEA.md §4.2).

    Es el artefacto central del proyecto: no una imagen, sino la **secuencia de
    decisiones** que la produce. ``final_image`` lo rellena N7 más tarde.
    """

    id: str
    prompt: str

    semantic_plan: SemanticPlan
    layout: SpatialLayout
    composition: Composition
    lighting: Lighting
    materials: Materials
    color_script: ColorScript

    visual_tokens: List[str] = Field(default_factory=list)

    # N7 — render (lo rellena el orquestador end-to-end; vacío tras solo N1–N6).
    # `final_image` = variación principal; `final_images` = las N variaciones.
    enriched_prompt: Optional[str] = None
    final_image: Optional[str] = None
    final_images: List[str] = Field(default_factory=list)
    render: Optional[StageTelemetry] = None

    telemetry: Dict[str, StageTelemetry] = Field(default_factory=dict)
    meta: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _totals_consistent(self) -> "VCoTTrace":
        # Invariante barato: si hay telemetría, debe cubrir solo etapas N1–N6.
        unknown = set(self.telemetry) - set(STAGE_MODELS)
        if unknown:
            raise ValueError(f"telemetría de etapas desconocidas: {sorted(unknown)}")
        return self

    def _render_compute_s(self) -> float:
        return self.render.compute_s if self.render else 0.0

    def _render_cost(self) -> float:
        return self.render.projected_cost_usd if self.render else 0.0

    @property
    def total_compute_s(self) -> float:
        """Tiempo de cómputo de razonamiento N1–N6 (sin render)."""
        return round(sum(t.compute_s for t in self.telemetry.values()), 6)

    @property
    def total_projected_cost_usd(self) -> float:
        """Coste del razonamiento N1–N6 (sin render)."""
        return round(sum(t.projected_cost_usd for t in self.telemetry.values()), 8)

    @property
    def e2e_compute_s(self) -> float:
        """Tiempo de cómputo de la cadena completa N1–N7 (con render si existe)."""
        return round(self.total_compute_s + self._render_compute_s(), 6)

    @property
    def e2e_cost_usd(self) -> float:
        """Coste de la cadena completa N1–N7 (razonamiento + render)."""
        return round(self.total_projected_cost_usd + self._render_cost(), 8)
