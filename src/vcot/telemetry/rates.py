"""Tarifas de Modal.com — única fuente de verdad (IDEA.md §8.1).

La unidad base es **$/segundo** porque la facturación de Modal es por uso real.
Las tarifas por hora se derivan (`* 3600`) y se exponen solo por conveniencia;
nunca se almacenan por separado para evitar que diverjan.

Si Modal actualiza precios, este archivo es el único punto a tocar: todos los
informes de coste del proyecto importan desde aquí.
"""

from __future__ import annotations

from typing import Mapping

#: $/segundo por GPU. Las claves usan la nomenclatura de Modal (`gpu="A100-80GB"`).
GPU_PER_SECOND: Mapping[str, float] = {
    "B200": 0.001736,
    "H200": 0.001261,
    "H100": 0.001097,
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}

#: $/segundo por núcleo físico de CPU (≈ 2 vCPU).
CPU_PER_SECOND: float = 0.0000131

#: $/segundo por GiB de memoria (se factura aparte del GPU).
MEM_PER_SECOND_PER_GIB: float = 0.00000222

#: $/GiB/mes de almacenamiento en Volume (1 TiB/mes gratis, no modelado aquí).
VOLUME_PER_GIB_MONTH: float = 0.09

#: Mínimo de núcleos facturados por contenedor.
MIN_CONTAINER_CORES: float = 0.125

#: Alias amigables → clave canónica de :data:`GPU_PER_SECOND`.
_GPU_ALIASES: Mapping[str, str] = {
    "A100": "A100-80GB",
    "A100-40": "A100-40GB",
    "A100-80": "A100-80GB",
    "RTX6000": "RTX-PRO-6000",
    "RTXPRO6000": "RTX-PRO-6000",
}


def _normalize_gpu(name: str) -> str:
    key = name.strip().upper().replace("_", "-")
    # Normaliza p.ej. "a100-80gb" → "A100-80GB" respetando el casing del catálogo.
    for canonical in GPU_PER_SECOND:
        if canonical.upper() == key:
            return canonical
    alias = _GPU_ALIASES.get(key) or _GPU_ALIASES.get(name.strip().upper())
    if alias is not None:
        return alias
    raise KeyError(
        f"GPU desconocida: {name!r}. Opciones: {', '.join(sorted(GPU_PER_SECOND))}"
    )


def gpu_rate(name: str, *, per: str = "second") -> float:
    """Tarifa de una GPU. ``per`` ∈ {"second", "hour"}.

    Acepta alias comunes (``"A100"`` → ``"A100-80GB"``) y es insensible a
    mayúsculas/guiones.
    """
    rate = GPU_PER_SECOND[_normalize_gpu(name)]
    return _scale(rate, per)


def resource_rate(
    gpu: str | None = None,
    *,
    cores: float = 0.0,
    mem_gib: float = 0.0,
    per: str = "second",
) -> float:
    """Tarifa combinada GPU + CPU + memoria de un contenedor.

    Aplica el mínimo de :data:`MIN_CONTAINER_CORES` cuando se especifican CPU.
    El resultado es lo que multiplica al tiempo de cómputo para obtener el coste.
    """
    rate = 0.0
    if gpu is not None:
        rate += GPU_PER_SECOND[_normalize_gpu(gpu)]
    if cores:
        rate += CPU_PER_SECOND * max(cores, MIN_CONTAINER_CORES)
    if mem_gib:
        rate += MEM_PER_SECOND_PER_GIB * mem_gib
    return _scale(rate, per)


def _scale(rate_per_second: float, per: str) -> float:
    if per == "second":
        return rate_per_second
    if per == "hour":
        return rate_per_second * 3600.0
    raise ValueError(f"`per` debe ser 'second' o 'hour', no {per!r}")
