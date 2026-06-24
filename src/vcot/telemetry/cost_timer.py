"""Cronómetro de coste embebido (IDEA.md §7.3).

Mide ``compute_s`` real de una etapa y lo convierte a USD usando :mod:`vcot.telemetry.rates`::

    from vcot.telemetry import cost_timer

    with cost_timer(gpu="H100") as t:
        image = run_flux(...)
    telemetry["render"] = {"compute_s": t.seconds, "cost_usd": t.cost}

El coste se calcula como ``compute_s × (rate_gpu + rate_cpu·cores + rate_mem·gib)``.
El cronómetro mide tiempo de pared con :func:`time.perf_counter`, que es lo que
factura Modal mientras el contenedor está vivo.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Optional, Type

from vcot.telemetry.rates import resource_rate


class CostTimer:
    """Context manager que mide tiempo de cómputo y su coste en USD.

    Parameters
    ----------
    gpu:
        Nombre de GPU de Modal (p.ej. ``"H100"``) o ``None`` para etapas sin GPU.
    cores:
        Núcleos físicos de CPU reservados (se aplica el mínimo del contenedor).
    mem_gib:
        Memoria reservada en GiB.

    Attributes
    ----------
    seconds:
        Duración medida (``compute_s``). Vale ``0.0`` hasta salir del bloque.
    cost:
        Coste en USD = ``seconds × rate``.
    """

    def __init__(
        self,
        gpu: Optional[str] = None,
        *,
        cores: float = 0.0,
        mem_gib: float = 0.0,
    ) -> None:
        self.gpu = gpu
        self.cores = cores
        self.mem_gib = mem_gib
        self.rate_per_second = resource_rate(gpu, cores=cores, mem_gib=mem_gib)
        self.seconds: float = 0.0
        self._start: Optional[float] = None

    @property
    def cost(self) -> float:
        """Coste acumulado en USD según el tiempo medido hasta ahora."""
        return self.elapsed * self.rate_per_second

    @property
    def elapsed(self) -> float:
        """Segundos transcurridos: en vivo dentro del bloque, congelado al salir."""
        if self._start is None:
            return self.seconds
        return time.perf_counter() - self._start

    def __enter__(self) -> "CostTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        assert self._start is not None
        self.seconds = time.perf_counter() - self._start
        self._start = None
        # No se suprime ninguna excepción: el coste del trabajo fallido también cuenta.

    def as_dict(self) -> dict[str, float]:
        """Telemetría serializable de la etapa (compatible con el registro de §4.2)."""
        return {
            "compute_s": round(self.seconds, 6),
            "rate_usd_per_s": self.rate_per_second,
            "cost_usd": round(self.cost, 8),
        }


def cost_timer(
    gpu: Optional[str] = None,
    *,
    cores: float = 0.0,
    mem_gib: float = 0.0,
) -> CostTimer:
    """Crea un :class:`CostTimer`. Pensado para usarse como context manager."""
    return CostTimer(gpu, cores=cores, mem_gib=mem_gib)
