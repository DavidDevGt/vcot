"""Semilla determinista por muestra (reproducibilidad bit-a-bit, IDEA.md §4.2).

Un dataset de research debe poder **regenerarse exactamente** desde el manifiesto.
Para eso cada muestra necesita una semilla estable y derivable de una clave (el
prompt), no aleatoria. ``derive_seed`` produce un entero de 32 bits determinista
que se pasa al planner (cadena N1–N6) y al renderer (N7), y queda registrado en
``meta.seed`` / ``images[].seed``.
"""

from __future__ import annotations

import hashlib

#: vLLM/torch aceptan semillas en uint32; acotamos a ese rango.
_SEED_MODULUS = 2**32


def derive_seed(key: str, *, salt: str = "vcot") -> int:
    """Entero determinista en ``[0, 2**32)`` derivado de ``key`` (hash estable)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _SEED_MODULUS
