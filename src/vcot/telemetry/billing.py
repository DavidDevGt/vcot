"""Coste **REAL facturado** por Modal — toda la vida del contenedor (IDEA.md §8).

`cost_timer` mide el coste *marginal* de una llamada: ``inferencia × tarifa GPU``.
Es lo correcto para "coste por imagen", pero **no es lo que pagás**. Modal factura
la vida completa del contenedor:

    arranque  +  carga del modelo (@modal.enter)  +  todas las inferencias
              +  el idle del `scaledown_window` antes de apagarse

…y suma **CPU y memoria** además de la GPU. Para corridas dispersas eso puede ser
~10–16× el coste marginal (carga + idle se pagan una vez por contenedor; se
amortizan solo si el contenedor cálido sirve muchas llamadas).

Dos herramientas, dos granularidades:

- :class:`ContainerMeter` — **medición real**. Se arranca en ``@modal.enter()`` y
  se cierra en ``@modal.exit()``; el tiempo de pared entre ambos es exactamente la
  ventana que Modal factura (incluye carga e idle tail). CPU y memoria se leen del
  **cgroup** del contenedor (uso real), porque Modal cobra ``max(reservado, usado)``
  y aquí no reservamos nada explícito.
- :func:`projected_container_cost` — **estimación desde el cliente**, que no ve el
  idle tail en vivo: ``vida ≈ carga + activo + scaledown_window``. Da una cifra
  realista de inmediato; la verdad exacta la deja el medidor en el Volume.

La tarifa sale de :mod:`vcot.telemetry.rates` (única fuente de verdad).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from vcot.telemetry.rates import (
    CPU_PER_SECOND,
    MEM_PER_SECOND_PER_GIB,
    MIN_CONTAINER_CORES,
    gpu_rate,
)

# --------------------------------------------------------------------------- #
# Lectura de uso real desde el cgroup del contenedor (Linux; Modal usa cgroup v2)
# --------------------------------------------------------------------------- #

_CPU_STAT_V2 = "/sys/fs/cgroup/cpu.stat"          # línea "usage_usec <N>"
_CPU_ACCT_V1 = "/sys/fs/cgroup/cpuacct/cpuacct.usage"  # nanosegundos
_MEM_PEAK_V2 = "/sys/fs/cgroup/memory.peak"       # bytes (pico)
_MEM_PEAK_V1 = "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"
_MEM_CUR_V2 = "/sys/fs/cgroup/memory.current"     # bytes (instantáneo, fallback)

_GIB = 1024 ** 3


def read_cpu_core_seconds() -> Optional[float]:
    """Núcleo-segundos de CPU consumidos por el contenedor, o ``None`` fuera de Linux.

    Es exactamente lo que Modal factura de CPU (``max`` con el mínimo reservado).
    """
    try:
        with open(_CPU_STAT_V2, encoding="ascii") as fh:
            for line in fh:
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) / 1_000_000.0
    except OSError:
        pass
    try:
        with open(_CPU_ACCT_V1, encoding="ascii") as fh:
            return int(fh.read().strip()) / 1_000_000_000.0
    except OSError:
        return None
    return None


def read_mem_peak_gib() -> Optional[float]:
    """Pico de memoria del contenedor en GiB, o ``None`` fuera de Linux."""
    for path in (_MEM_PEAK_V2, _MEM_PEAK_V1, _MEM_CUR_V2):
        try:
            with open(path, encoding="ascii") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw and raw != "max":
            return int(raw) / _GIB
    return None


# --------------------------------------------------------------------------- #
# Coste real de un contenedor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContainerCost:
    """Desglose del coste real (o proyectado) de un contenedor Modal."""

    gpu: Optional[str]
    billed_s: float          # vida facturada del contenedor (carga + activo + idle)
    gpu_cost_usd: float
    cpu_core_s: float        # núcleo-segundos facturados (≥ mínimo reservado)
    cpu_cost_usd: float
    mem_gib: float           # memoria facturada (pico medido o estimación)
    mem_cost_usd: float
    real_cost_usd: float     # GPU + CPU + memoria
    measured: bool           # True si CPU/mem salen del cgroup; False si es estimación

    def as_dict(self) -> dict:
        return {
            "gpu": self.gpu,
            "billed_s": round(self.billed_s, 3),
            "gpu_cost_usd": round(self.gpu_cost_usd, 8),
            "cpu_core_s": round(self.cpu_core_s, 3),
            "cpu_cost_usd": round(self.cpu_cost_usd, 8),
            "mem_gib": round(self.mem_gib, 3),
            "mem_cost_usd": round(self.mem_cost_usd, 8),
            "real_cost_usd": round(self.real_cost_usd, 8),
            "measured": self.measured,
        }


def _compose(
    gpu: Optional[str],
    billed_s: float,
    cpu_core_s: float,
    mem_gib: float,
    *,
    measured: bool,
) -> ContainerCost:
    """Compone el coste a partir de la vida facturada y el uso de CPU/memoria.

    - **GPU**: ``billed_s × tarifa`` — exacto (la GPU está reservada toda la vida).
    - **CPU**: ``max(usado, mínimo_reservado × billed_s) × tarifa`` — Modal cobra
      ``max(reservado, usado)`` y el contenedor reserva siempre ``MIN_CONTAINER_CORES``.
    - **Memoria**: ``mem_gib × billed_s × tarifa`` — usando el **pico** como cota
      superior conservadora del integral memoria·segundo (la RAM crece en la carga).
    """
    gpu_cost = billed_s * (gpu_rate(gpu) if gpu else 0.0)
    cpu_core_s = max(cpu_core_s, MIN_CONTAINER_CORES * billed_s)
    cpu_cost = cpu_core_s * CPU_PER_SECOND
    mem_cost = mem_gib * billed_s * MEM_PER_SECOND_PER_GIB
    return ContainerCost(
        gpu=gpu,
        billed_s=billed_s,
        gpu_cost_usd=gpu_cost,
        cpu_core_s=cpu_core_s,
        cpu_cost_usd=cpu_cost,
        mem_gib=mem_gib,
        mem_cost_usd=mem_cost,
        real_cost_usd=gpu_cost + cpu_cost + mem_cost,
        measured=measured,
    )


class ContainerMeter:
    """Mide el coste real de un contenedor Modal de ``@modal.enter`` a ``@modal.exit``.

    Construilo como **primera** acción del ``@modal.enter()`` (antes de cargar el
    modelo) y llamá :meth:`stop` en el ``@modal.exit()``::

        @modal.enter()
        def load(self):
            self.meter = ContainerMeter(GPU)
            ...  # cargar pesos

        @modal.exit()
        def _bill(self):
            cost = self.meter.stop()
            print(cost.real_cost_usd)

    Parameters
    ----------
    gpu:
        Nombre de GPU de Modal, o ``None`` para contenedores sin GPU.
    mem_gib_fallback:
        Memoria a facturar si el cgroup no es legible (tests/local).
    """

    def __init__(
        self,
        gpu: Optional[str],
        *,
        mem_gib_fallback: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
        cpu_reader: Callable[[], Optional[float]] = read_cpu_core_seconds,
        mem_reader: Callable[[], Optional[float]] = read_mem_peak_gib,
    ) -> None:
        self.gpu = gpu
        self.mem_gib_fallback = mem_gib_fallback
        self._clock = clock
        self._cpu_reader = cpu_reader
        self._mem_reader = mem_reader
        self._t0 = clock()
        self._cpu0 = cpu_reader()  # línea base de núcleo-segundos (o None)
        self.result: Optional[ContainerCost] = None

    def stop(self) -> ContainerCost:
        """Cierra el medidor y devuelve (y memoiza) el coste real del contenedor."""
        billed_s = self._clock() - self._t0
        cpu_now = self._cpu_reader()
        if cpu_now is not None and self._cpu0 is not None:
            cpu_core_s = max(cpu_now - self._cpu0, 0.0)
        else:
            cpu_core_s = 0.0  # _compose lo eleva al mínimo reservado
        mem_peak = self._mem_reader()
        measured = mem_peak is not None
        mem_gib = mem_peak if measured else self.mem_gib_fallback
        self.result = _compose(
            self.gpu, billed_s, cpu_core_s, mem_gib, measured=measured
        )
        return self.result


def projected_container_cost(
    *,
    gpu: Optional[str],
    active_s: float,
    model_load_s: float = 0.0,
    scaledown_window: float = 0.0,
    cpu_core_s: float = 0.0,
    mem_gib: float = 0.0,
) -> ContainerCost:
    """Estimación de coste real desde el cliente (no ve el idle tail en vivo).

    ``vida ≈ carga + activo + scaledown_window``. Pensada para imprimir una cifra
    realista en cuanto vuelve la llamada; la verdad exacta la mide
    :class:`ContainerMeter` dentro del contenedor.
    """
    billed_s = model_load_s + active_s + scaledown_window
    return _compose(gpu, billed_s, cpu_core_s, mem_gib, measured=False)
