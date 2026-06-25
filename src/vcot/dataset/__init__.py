"""Dataset de *pensamiento visual* (IDEA.md §4).

No imágenes: la **secuencia de decisiones**. Aquí viven los prompts semilla y la
conversión de una traza a ejemplos de entrenamiento para destilar Klein (§5.1).
La generación masiva (fan-out) corre sobre Modal en ``modal_app/dataset.py``.
"""

from __future__ import annotations

from vcot.dataset.seed_prompts import SEED_PROMPTS
from vcot.dataset.sft import trace_to_sft, trace_to_token_target

__all__ = ["SEED_PROMPTS", "trace_to_sft", "trace_to_token_target"]
