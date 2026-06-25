"""Destilación de Klein (IDEA.md §5, M5).

Convierte el dataset de trazas V-CoT en ejemplos de entrenamiento y expone un
entrenamiento SFT. La preparación de datos es stdlib puro y corre/se testea en
local; el entrenamiento real requiere GPU + el extra ``train`` (TRL/transformers)
y está pensado para lanzarse sobre Modal.
"""

from __future__ import annotations

from vcot.train.distill import build_sft_dataset

__all__ = ["build_sft_dataset"]
