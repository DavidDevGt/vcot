"""Traza V-CoT → ejemplos de entrenamiento para destilar Klein (IDEA.md §5.1).

Dos formatos de objetivo de destilación:

- :func:`trace_to_sft`: ejemplo de chat (``messages``) donde el assistant produce
  el **razonamiento completo** (N1–N6) + visual tokens + prompt de render. Enseña
  al modelo pequeño a *pensar*, no solo a renderizar.
- :func:`trace_to_token_target`: par ``prompt → secuencia de Visual Tokens``, la
  representación compacta de §2.2 (útil para cabezas autoregresivas / MoE).
"""

from __future__ import annotations

import json
from typing import Dict, List

from vcot.pipeline.enrich import enrich_prompt
from vcot.pipeline.prompts import SYSTEM_PROMPT
from vcot.pipeline.schemas import STAGE_MODELS, VCoTTrace


def _reasoning_dict(trace: VCoTTrace) -> Dict[str, dict]:
    return {stage: getattr(trace, stage).model_dump() for stage in STAGE_MODELS}


def trace_to_sft(trace: VCoTTrace) -> Dict[str, List[dict]]:
    """Ejemplo SFT estilo chat: prompt → cadena de razonamiento completa."""
    render_prompt = trace.enriched_prompt or enrich_prompt(trace)
    assistant = json.dumps(
        {
            "reasoning": _reasoning_dict(trace),
            "visual_tokens": trace.visual_tokens,
            "render_prompt": render_prompt,
        },
        ensure_ascii=False,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": trace.prompt},
            {"role": "assistant", "content": assistant},
        ]
    }


def trace_to_token_target(trace: VCoTTrace) -> Dict[str, str]:
    """Par compacto ``prompt → visual tokens`` (§2.2)."""
    return {"prompt": trace.prompt, "completion": " ".join(trace.visual_tokens)}
