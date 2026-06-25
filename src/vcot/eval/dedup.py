"""Dedup perceptual del dataset (IDEA.md §4, calidad research-grade).

Detecta imágenes casi-duplicadas con **pHash (DCT)** de 64 bits y distancia de
Hamming. pHash mira la estructura de baja frecuencia (vía DCT-II), así que es
robusto a cambios de brillo/contraste/escala — a diferencia de aHash, que se
deja engañar por un simple cambio de exposición. La matemática (``hamming``,
``duplicate_indices``) es pura y testeable; ``phash_dct`` hace lazy-import de
Pillow solo al decodificar una imagen real.

Se deduplica por **contenido** (no por prompt): dos renders distintos que salieron
visualmente idénticos cuentan como duplicado y se marcan para excluir del gate.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set

#: Umbral de Hamming por defecto para pHash de 64 bits (≤ ⇒ duplicado).
DEFAULT_THRESHOLD = 8


def hamming(a: int, b: int) -> int:
    """Distancia de Hamming entre dos hashes enteros."""
    return (a ^ b).bit_count()


def _dct_1d(vector: Sequence[float]) -> List[float]:
    """DCT-II 1D (referencia, sin numpy). O(n²); n pequeño (32) ⇒ barato."""
    n = len(vector)
    result = []
    factor = math.pi / n
    for k in range(n):
        total = 0.0
        for i, v in enumerate(vector):
            total += v * math.cos((i + 0.5) * k * factor)
        result.append(total)
    return result


def _load_gray(image, size: int):
    """Carga y convierte a gris ``size×size`` (ruta, bytes o PIL.Image)."""
    from io import BytesIO

    from PIL import Image  # lazy

    if isinstance(image, (bytes, bytearray)):
        img = Image.open(BytesIO(bytes(image)))
    elif isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    return img.convert("L").resize((size, size), Image.Resampling.LANCZOS)


def phash_dct(image, *, hash_size: int = 8, img_size: int = 32) -> int:
    """pHash de ``hash_size**2`` bits: DCT-II 2D y umbral por la mediana del bloque.

    Toma el bloque de baja frecuencia ``hash_size×hash_size`` (esquina superior
    izquierda de la DCT, excluyendo el término DC) y pone a 1 los coeficientes por
    encima de la mediana. Determinista.
    """
    img = _load_gray(image, img_size)
    pixels = list(img.getdata())
    rows = [pixels[r * img_size:(r + 1) * img_size] for r in range(img_size)]

    # DCT 2D = DCT por filas y luego por columnas.
    dct_rows = [_dct_1d(row) for row in rows]
    dct = [
        _dct_1d([dct_rows[r][c] for r in range(img_size)])
        for c in range(img_size)
    ]  # dct[c][r]: transpuesta, da igual para el bloque de baja frecuencia (simétrico)

    low = [dct[c][r] for r in range(hash_size) for c in range(hash_size)]
    # La mediana se calcula excluyendo el coeficiente DC (domina y no aporta señal).
    rest = sorted(low[1:])
    median = rest[len(rest) // 2] if rest else 0.0
    bits = 0
    for i, coef in enumerate(low):
        if coef > median:
            bits |= 1 << i
    return bits


def average_hash(image, *, hash_size: int = 8) -> int:
    """aHash de ``hash_size**2`` bits (legacy; ``phash_dct`` es el recomendado)."""
    img = _load_gray(image, hash_size)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << i
    return bits


def duplicate_indices(hashes: Iterable[int], *, threshold: int = DEFAULT_THRESHOLD) -> Set[int]:
    """Índices que duplican (Hamming ≤ ``threshold``) a uno anterior conservado.

    Se conserva el primero de cada grupo; los siguientes se marcan duplicados.
    """
    kept: List[int] = []
    dups: Set[int] = set()
    for i, h in enumerate(hashes):
        if any(hamming(h, k) <= threshold for k in kept):
            dups.add(i)
        else:
            kept.append(h)
    return dups
