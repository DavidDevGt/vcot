"""Instrumentación de inferencia y coste (IDEA.md §6–§8).

`rates` es la **única fuente de verdad** de las tarifas de Modal; `cost_timer`
mide el tiempo de cómputo real de cada etapa y lo convierte a USD usando esas
tarifas.
"""

from __future__ import annotations

from vcot.telemetry.cost_timer import CostTimer, cost_timer
from vcot.telemetry.rates import (
    CPU_PER_SECOND,
    GPU_PER_SECOND,
    MEM_PER_SECOND_PER_GIB,
    MIN_CONTAINER_CORES,
    VOLUME_PER_GIB_MONTH,
    gpu_rate,
    resource_rate,
)

__all__ = [
    "CostTimer",
    "cost_timer",
    "gpu_rate",
    "resource_rate",
    "GPU_PER_SECOND",
    "CPU_PER_SECOND",
    "MEM_PER_SECOND_PER_GIB",
    "VOLUME_PER_GIB_MONTH",
    "MIN_CONTAINER_CORES",
]
