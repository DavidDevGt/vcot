"""Cliente de LLM **local** para el planner (IDEA.md §4.1 opción B).

Hablamos con el modelo vía un endpoint **OpenAI-compatible**, que es el estándar
de facto de Ollama, vLLM, LM Studio y llama.cpp. Ventaja de diseño: el mismo
cliente sirve hoy en local (Ollama) y mañana en Modal (vLLM self-hosted) — solo
cambia ``base_url``. No usamos ninguna API en la nube.

El transporte es **stdlib pura** (``urllib``): el núcleo del proyecto no añade
dependencias de runtime por esto.

Configuración por variables de entorno (todas opcionales)::

    VCOT_LLM_BASE_URL   # default http://localhost:11434/v1  (Ollama)
    VCOT_LLM_MODEL      # default qwen3:8b
    VCOT_LLM_API_KEY    # default "local"  (los runtimes locales lo ignoran)

Nota Qwen3: es un modelo de razonamiento con *thinking mode* activado por defecto;
en el path local puede emitir un bloque ``<think>…</think>`` (el planner lo
descarta al parsear). Para suprimirlo, añade ``/no_think`` al prompt o usa la
opción del runtime (en Ollama, ``think: false``).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Protocol, runtime_checkable

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:8b"


@dataclass
class LLMResponse:
    """Respuesta del LLM + uso de tokens (para telemetría de §6)."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class LLMClient(Protocol):
    """Contrato mínimo del planner. Cualquier backend que lo cumpla encaja."""

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> LLMResponse:
        ...


class LocalLLMClient:
    """Cliente para un servidor LLM local OpenAI-compatible."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        temperature: float = 0.4,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("VCOT_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("VCOT_LLM_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("VCOT_LLM_API_KEY", "local")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "stream": False,
        }
        if json_mode:
            # Soportado por Ollama, vLLM, LM Studio y llama.cpp. Si el backend no
            # lo soporta, el prompt ya pide JSON y la validación de pydantic +
            # reintento lo cubre igualmente.
            body["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"No se pudo contactar el LLM local en {self.base_url}. "
                f"¿Está arrancado el runtime (p.ej. `ollama serve`) y el modelo "
                f"{self.model!r} disponible? Detalle: {exc}"
            ) from exc

        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


class FakeLLMClient:
    """Cliente determinista para tests (sin red).

    Devuelve las respuestas en ``responses`` en orden. Cada elemento puede ser un
    str (se envía tal cual) o un dict (se serializa a JSON). Útil para simular
    tanto salidas válidas como inválidas (para probar el bucle de reintento).
    """

    def __init__(self, responses: List[object]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.calls: List[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = True) -> LLMResponse:
        self.calls.append((system, user))
        if self._i >= len(self._responses):
            raise AssertionError("FakeLLMClient se quedó sin respuestas")
        item = self._responses[self._i]
        self._i += 1
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(
            text=text,
            input_tokens=max(1, len(system) + len(user)) // 4,
            output_tokens=max(1, len(text)) // 4,
        )
