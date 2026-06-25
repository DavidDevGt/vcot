"""Orquestación end-to-end de la cadena V-CoT N1→N7 (IDEA.md §2, §3).

Compone el razonamiento (planner N1–N6) con el render (N7), pasando por el
*enriquecimiento* de prompt (§3.1). Es agnóstico del backend: recibe un planner y
un ``render_fn`` como callables, de modo que la lógica se prueba con dobles y la
versión real vive en ``modal_app/pipeline.py`` (ambos sobre Modal).

El resultado es la :class:`VCoTTrace` **completa**: razonamiento + prompt
enriquecido + imagen final + coste de cada etapa (N1–N7).
"""

from __future__ import annotations

from typing import Callable, Protocol

from vcot.pipeline.enrich import enrich_prompt
from vcot.pipeline.schemas import ImageRef, StageTelemetry, VCoTTrace


class SupportsPlan(Protocol):
    def plan(self, prompt: str) -> VCoTTrace:
        ...


#: ``render_fn(enriched_prompt, sample_id) -> render record`` con la forma que
#: produce ``modal_app/renderer.py``: ``{"final_image", "images": [...],
#: "telemetry": {"render": {...}}, "meta": {"gpu": ...}}``. Se le pasa el
#: ``sample_id`` (= ``trace.id``) para que las imágenes queden ligadas a la traza.
RenderFn = Callable[[str, str], dict]


def _coerce_render_telemetry(record: dict) -> StageTelemetry:
    """Adapta la telemetría del renderer (``cost_usd``/``gpu``) a StageTelemetry."""
    tele = record.get("telemetry", {}).get("render", {})
    gpu = record.get("meta", {}).get("gpu", "unknown")
    return StageTelemetry(
        compute_s=tele.get("compute_s", 0.0),
        rate_usd_per_s=tele.get("rate_usd_per_s", 0.0),
        projected_cost_usd=tele.get("cost_usd", tele.get("projected_cost_usd", 0.0)),
        projected_gpu=gpu,
    )


def run_pipeline(
    prompt: str,
    planner: SupportsPlan,
    render_fn: RenderFn,
    *,
    enrich: Callable[[VCoTTrace], str] = enrich_prompt,
) -> VCoTTrace:
    """Ejecuta N1→N7 y devuelve la traza completa con imagen y coste e2e."""
    trace = planner.plan(prompt)
    trace.enriched_prompt = enrich(trace)

    record = render_fn(trace.enriched_prompt, trace.id)
    trace.final_image = record.get("final_image")
    trace.final_images = record.get("final_images") or (
        [trace.final_image] if trace.final_image else []
    )
    trace.images = [ImageRef.model_validate(im) for im in record.get("images", [])]
    trace.render = _coerce_render_telemetry(record)
    return trace
