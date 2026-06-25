"""Análisis de inferencia y coste del dataset V-CoT (IDEA.md §6, §8, §9).

Lee los registros de traza (``traces.jsonl``) y produce un informe de coste y
latencia por etapa: medias, percentiles y proyecciones a 1k/1M muestras. Stdlib
puro (sin pandas) para que sea trivial de correr y testear.
"""

from __future__ import annotations

from vcot.analysis.aggregate import format_report, load_jsonl, summarize, write_csv

__all__ = ["load_jsonl", "summarize", "format_report", "write_csv"]
