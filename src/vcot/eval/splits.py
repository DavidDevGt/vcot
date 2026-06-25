"""Particionado train/val/test determinista y sin fuga (IDEA.md §5, §8).

Asigna cada muestra a un split por **hash de una clave estable** (no por shuffle
global), de modo que:

- es **reproducible** (misma clave + misma semilla ⇒ mismo split),
- evita **fuga**: muestras que comparten clave (p.ej. el mismo prompt con varias
  semillas de render) caen siempre en el mismo split.

Función pura (stdlib). El caller pasa como clave el ``prompt`` para agrupar todas
las variaciones de un mismo brief.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Tuple

Split = str  # "train" | "val" | "test"
Ratios = Tuple[float, float, float]
DEFAULT_RATIOS: Ratios = (0.8, 0.1, 0.1)


def split_fraction(key: str, *, seed: int = 0) -> float:
    """Fracción estable en ``[0,1)`` derivada de ``key`` (hash → uniforme)."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) % 1_000_000) / 1_000_000


def assign_split(key: str, *, ratios: Ratios = DEFAULT_RATIOS, seed: int = 0) -> Split:
    """Asigna ``key`` a 'train'/'val'/'test' según ``ratios`` (suman 1)."""
    train, val, test = ratios
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"ratios deben sumar 1.0: {ratios}")
    f = split_fraction(key, seed=seed)
    if f < train:
        return "train"
    if f < train + val:
        return "val"
    return "test"


def assign_splits(
    keys: Iterable[str], *, ratios: Ratios = DEFAULT_RATIOS, seed: int = 0
) -> Dict[str, Split]:
    """Mapa ``key → split``. Claves repetidas ⇒ mismo split (sin fuga)."""
    return {key: assign_split(key, ratios=ratios, seed=seed) for key in set(keys)}
