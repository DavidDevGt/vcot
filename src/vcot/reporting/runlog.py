"""Ledger de ejecuciones — todo proceso queda registrado (IDEA.md §6).

Cada corrida (planner, render, dataset, pipeline) escribe una línea en un JSONL
append-only. El informe final (``vcot.reporting.report``) lo incluye como sección
de auditoría. Stdlib puro.

Uso típico desde un entrypoint::

    from vcot.reporting.runlog import track_run

    with track_run("outputs/runs.jsonl", kind="dataset", model=repo, gpu=gpu) as run:
        ...  # hacer el trabajo
        run["n_items"] = n
        run["total_cost_usd"] = total

El context manager cronometra, marca ``status`` ("ok"/"error"), captura la
excepción (la re-lanza) y escribe la línea al salir.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional


@dataclass
class RunRecord:
    """Una ejecución registrada en el ledger."""

    kind: str  # "planner" | "renderer" | "dataset" | "pipeline"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    duration_s: float = 0.0
    model: Optional[str] = None
    gpu: Optional[str] = None
    n_items: int = 0
    total_cost_usd: float = 0.0
    status: str = "ok"
    error: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)


def append_run(path: str, record: Dict[str, object]) -> None:
    """Añade una línea al ledger (crea el archivo y carpeta si hace falta)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_runs(path: str) -> List[dict]:
    """Carga el ledger; devuelve ``[]`` si no existe."""
    if not os.path.exists(path):
        return []
    runs: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


@contextmanager
def track_run(path: str, kind: str, **fields) -> Iterator[dict]:
    """Context manager que cronometra y registra una ejecución en el ledger.

    Cede un dict mutable; rellená ``n_items``/``total_cost_usd`` dentro del bloque.
    """
    record = asdict(RunRecord(kind=kind, **fields))
    t0 = time.perf_counter()
    try:
        yield record
    except BaseException as exc:  # noqa: BLE001 - registramos cualquier fallo
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["duration_s"] = round(time.perf_counter() - t0, 3)
        append_run(path, record)
