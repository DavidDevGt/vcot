"""Pipeline de razonamiento V-CoT (IDEA.md §2).

Las etapas N1–N6 (el *pensamiento visual*) como esquemas validables, el planner
que las genera con un LLM local, y la serialización a Visual Tokens. N7 (render)
vive en ``modal_app/``.
"""

from __future__ import annotations

from vcot.pipeline.enrich import enrich_prompt
from vcot.pipeline.llm import FakeLLMClient, LLMClient, LLMResponse, LocalLLMClient
from vcot.pipeline.pipeline import run_pipeline
from vcot.pipeline.planner import Planner, PlannerError
from vcot.pipeline.schemas import (
    STAGE_LABELS,
    STAGE_MODELS,
    ColorScript,
    Composition,
    DatasetMeta,
    Entity,
    ImageRef,
    Lighting,
    Materials,
    Relation,
    SemanticPlan,
    SpatialLayout,
    StageTelemetry,
    VCoTTrace,
)
from vcot.pipeline.visual_tokens import to_visual_tokens

__all__ = [
    "Planner",
    "PlannerError",
    "run_pipeline",
    "enrich_prompt",
    "LLMClient",
    "LLMResponse",
    "LocalLLMClient",
    "FakeLLMClient",
    "to_visual_tokens",
    "VCoTTrace",
    "ImageRef",
    "DatasetMeta",
    "StageTelemetry",
    "SemanticPlan",
    "SpatialLayout",
    "Entity",
    "Relation",
    "Composition",
    "Lighting",
    "Materials",
    "ColorScript",
    "STAGE_MODELS",
    "STAGE_LABELS",
]
