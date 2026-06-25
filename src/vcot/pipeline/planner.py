"""Planner V-CoT: genera la cadena de razonamiento N1–N6 (IDEA.md §2, §6).

El planner es el **corazón de la tesis**: en vez de saltar a la imagen, produce
la *secuencia de decisiones* que la precede, una etapa a la vez, cada una
condicionada por las anteriores. Cada etapa:

1. se genera con un LLM local (cualquier backend OpenAI-compatible),
2. se valida contra su esquema pydantic (la propiedad "criticable" de §5.3),
   con reintento automático si no valida,
3. se cronometra con ``cost_timer`` para proyectar su coste en una GPU de Modal
   (atando con ``vcot.telemetry.rates``), aunque corra en local.

El resultado es un :class:`~vcot.pipeline.schemas.VCoTTrace`: el dataset de
*pensamiento visual*, no de imágenes.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Dict

from pydantic import ValidationError

from vcot.pipeline.llm import LLMClient
from vcot.pipeline.prompts import SYSTEM_PROMPT, build_stage_prompt, repair_prompt
from vcot.pipeline.schemas import (
    STAGE_LABELS,
    STAGE_MODELS,
    StageTelemetry,
    VCoTTrace,
)
from vcot.pipeline.visual_tokens import to_visual_tokens
from vcot.telemetry import cost_timer, gpu_rate

DEFAULT_PROJECTED_GPU = "A100-40GB"  # referencia del planner en IDEA.md §8.3

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
# Modelos de razonamiento (p.ej. Qwen3) pueden emitir un bloque <think>…</think>
# antes de la respuesta. Lo quitamos para que no contamine el JSON de la etapa.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class PlannerError(RuntimeError):
    """Una etapa no produjo JSON válido tras agotar los reintentos."""


def _extract_json(text: str) -> dict:
    """Extrae el objeto JSON de la respuesta del LLM, tolerante a ruido."""
    s = _THINK_RE.sub("", text).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Quitar fences de markdown si los hubiera.
    s = _FENCE_RE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Último recurso: subcadena entre la primera '{' y la última '}'.
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError(f"sin JSON parseable en la respuesta: {text[:200]!r}")


class Planner:
    """Genera trazas V-CoT etapa por etapa con un LLM local.

    Parameters
    ----------
    client:
        Cualquier :class:`~vcot.pipeline.llm.LLMClient` (local por defecto).
    projected_gpu:
        GPU de Modal sobre la que se *proyecta* el coste (clave de ``rates``).
    max_retries:
        Reintentos por etapa si la salida no valida contra el esquema.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        projected_gpu: str = DEFAULT_PROJECTED_GPU,
        max_retries: int = 2,
    ) -> None:
        self.client = client
        self.projected_gpu = projected_gpu
        self.rate = gpu_rate(projected_gpu)  # valida la GPU al construir
        self.max_retries = max_retries

    def plan(self, prompt: str, *, sample_id: str | None = None) -> VCoTTrace:
        """Ejecuta la cadena completa N1–N6 y devuelve la traza."""
        sample_id = sample_id or uuid.uuid4().hex
        prior: Dict[str, dict] = {}
        stages: Dict[str, object] = {}
        telemetry: Dict[str, StageTelemetry] = {}

        for stage, model in STAGE_MODELS.items():
            schema = model.model_json_schema()
            base_user = build_stage_prompt(stage, prompt, prior, schema)
            obj, tele = self._run_stage(stage, model, base_user)
            stages[stage] = obj
            prior[stage] = obj.model_dump()
            telemetry[stage] = tele

        trace = VCoTTrace(
            id=sample_id,
            prompt=prompt,
            telemetry=telemetry,
            meta={
                "planner": getattr(self.client, "model", "local"),
                "base_url": getattr(self.client, "base_url", None),
                "projected_gpu": self.projected_gpu,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            **stages,  # type: ignore[arg-type]
        )
        trace.visual_tokens = to_visual_tokens(trace)
        return trace

    def _run_stage(self, stage: str, model, base_user: str):
        """Genera una etapa con reintentos; acumula coste y tokens de todos los intentos."""
        total_s = 0.0
        in_tok = out_tok = retries = 0
        user = base_user
        last_error = ""

        for attempt in range(self.max_retries + 1):
            with cost_timer(gpu=self.projected_gpu) as t:
                resp = self.client.complete(SYSTEM_PROMPT, user)
            total_s += t.seconds
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens

            try:
                obj = model.model_validate(_extract_json(resp.text))
                break
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                if attempt == self.max_retries:
                    raise PlannerError(
                        f"etapa {STAGE_LABELS[stage]} ({stage}) no validó tras "
                        f"{self.max_retries} reintentos: {last_error}"
                    ) from exc
                retries += 1
                user = (
                    base_user
                    + "\n\nPREVIOUS ANSWER:\n"
                    + resp.text
                    + "\n\n"
                    + repair_prompt(last_error)
                )

        # Se guardan valores en crudo (el redondeo es cosa de presentación, en el
        # CLI); así el invariante projected_cost == compute_s × rate es exacto.
        tele = StageTelemetry(
            compute_s=total_s,
            rate_usd_per_s=self.rate,
            projected_cost_usd=self.rate * total_s,
            projected_gpu=self.projected_gpu,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tokens_per_s=(out_tok / total_s) if total_s > 0 else 0.0,
            retries=retries,
            last_error=(last_error or None) if retries else None,
        )
        return obj, tele
