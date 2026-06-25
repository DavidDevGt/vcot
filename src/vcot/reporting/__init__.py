"""Reporting profesional del pipeline V-CoT (IDEA.md §6, §8, §9).

Dos artefactos para que **todo quede registrado**:

- :mod:`vcot.reporting.runlog` — un *ledger* append-only (``runs.jsonl``) donde
  cada proceso (planner, render, dataset, pipeline) deja constancia: cuándo,
  qué modelo/GPU, cuántos ítems, cuánto costó y si terminó bien.
- :mod:`vcot.reporting.report` — genera el **informe final** a partir de las
  trazas y el ledger: un bundle con fecha en ``reports/<timestamp>/`` en Markdown
  (humano), JSON (máquina) y CSV (por etapa).
"""

from __future__ import annotations

from vcot.reporting.report import build_report, render_markdown, write_bundle
from vcot.reporting.runlog import RunRecord, append_run, load_runs, track_run

__all__ = [
    "build_report",
    "render_markdown",
    "write_bundle",
    "RunRecord",
    "append_run",
    "load_runs",
    "track_run",
]
