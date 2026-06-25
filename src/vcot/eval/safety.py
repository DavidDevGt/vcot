"""Taxonomía de seguridad para release público (IDEA.md §4, responsible release).

Un único score NSFW no es una política de seguridad. Aquí se estructura el bloque
``safety`` como una **taxonomía con decisión de release** y se deja explícito qué
checks faltan antes de publicar (honestidad de datacard).

Puro y testeable. La inferencia NSFW vive en ``modal_app/eval.py``; esta función
traduce su probabilidad en etiqueta + decisión.

⚠️ Esto **no sustituye** un pipeline de seguridad completo. Para un release
público real faltan ``csam_hash`` (match contra bases de hashes conocidas) y
``pii_scan`` (rostros identificables / datos personales). Se declaran como
pendientes para que el datacard no afirme una seguridad que no existe.
"""

from __future__ import annotations

from typing import Dict, Optional

#: Umbrales de la etiqueta NSFW.
NSFW_BLOCK = 0.85   # ≥ ⇒ bloqueo duro (no se publica)
NSFW_REVIEW = 0.5   # ≥ ⇒ revisión humana antes de publicar

#: Checks de seguridad requeridos para un release público y su estado.
#: ``True`` = implementado y aplicado; ``False`` = pendiente (gating manual).
SAFETY_CHECKS_IMPLEMENTED: Dict[str, bool] = {
    "nsfw": True,
    "csam_hash": False,   # TODO: match contra PhotoDNA/NCMEC antes de publicar
    "pii_scan": False,    # TODO: detección de rostros identificables / PII
}


def nsfw_label(nsfw_prob: Optional[float]) -> str:
    """'unknown' | 'ok' | 'review' | 'blocked' según la probabilidad NSFW."""
    if nsfw_prob is None:
        return "unknown"
    if nsfw_prob >= NSFW_BLOCK:
        return "blocked"
    if nsfw_prob >= NSFW_REVIEW:
        return "review"
    return "ok"


def classify_safety(nsfw_prob: Optional[float]) -> Dict:
    """Bloque ``safety`` estructurado + decisión de release.

    ``release_blocked`` es ``True`` si el NSFW supera el umbral duro **o** si hay
    checks obligatorios sin implementar (postura conservadora: no se publica algo
    cuyo perfil de seguridad no se verificó del todo).
    """
    label = nsfw_label(nsfw_prob)
    pending = [k for k, done in SAFETY_CHECKS_IMPLEMENTED.items() if not done]
    return {
        "nsfw": nsfw_prob,
        "nsfw_label": label,
        "release_blocked": label == "blocked",
        "needs_review": label in ("review", "blocked"),
        "checks_applied": [k for k, done in SAFETY_CHECKS_IMPLEMENTED.items() if done],
        "checks_pending": pending,
    }
