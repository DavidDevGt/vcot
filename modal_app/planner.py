"""N1–N6 — Planner V-CoT sobre Modal.com (IDEA.md §4.1 opción B, §7).

Ejecuta la cadena de razonamiento en **Modal serverless GPU**, con un LLM
open-weights **self-hosted con vLLM dentro del contenedor** (no una API de nube).
Reutiliza tal cual la lógica de `vcot.pipeline.Planner`: solo cambia el cliente
HTTP local por uno que habla con el motor vLLM in-process.

Como corre en Modal, el `cost_timer` mide el **coste real** de GPU por etapa (no
proyectado): mismo cálculo `compute_s × rate`, pero aquí la GPU se factura de
verdad. Mismo patrón que `modal_app/renderer.py` (`@app.cls` + `@enter`).

Por defecto **Qwen3-8B** en A100-40GB (referencia del planner en IDEA.md §8.3).
Cambiar con VCOT_PLANNER_MODEL / VCOT_PLANNER_GPU.

Qwen3 es un modelo *híbrido de razonamiento*: se ejecuta con **thinking mode
desactivado** (`enable_thinking=False`) a propósito — en V-CoT la descomposición
N1–N6 ya ES el razonamiento explícito, así que no queremos un bloque `<think>`
opaco (que además dispara tokens y coste). Requiere **vLLM ≥ 0.8.5**.

Uso:
    modal run modal_app/planner.py --prompt "a lone astronaut in a gothic cathedral"
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

import modal

# --------------------------------------------------------------------------- #
# Configuración de modelo
# --------------------------------------------------------------------------- #

MODELS = {
    # 8B (dense, híbrido de razonamiento): ~16 GB en bf16, cabe holgado en A100-40GB.
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B",
        "gpu": "A100-40GB",
        "max_model_len": 8192,
    },
    # 14B: más calidad de traza; necesita A100-80GB.
    "qwen3-14b": {
        "repo": "Qwen/Qwen3-14B",
        "gpu": "A100-80GB",
        "max_model_len": 8192,
    },
}

MODEL = os.environ.get("VCOT_PLANNER_MODEL", "qwen3-8b")
if MODEL not in MODELS:
    raise SystemExit(f"VCOT_PLANNER_MODEL={MODEL!r} no válido. Opciones: {', '.join(MODELS)}")
CFG = MODELS[MODEL]
GPU = os.environ.get("VCOT_PLANNER_GPU", CFG["gpu"])  # debe ser una clave de rates.py

CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

#: Segundos que Modal mantiene vivo (y factura) el contenedor tras la última
#: llamada — coste de GPU puro que el `cost_timer` marginal no ve.
SCALEDOWN_WINDOW = 120

# --------------------------------------------------------------------------- #
# Imagen del contenedor
# --------------------------------------------------------------------------- #

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        # vLLM trae torch + CUDA runtime compatibles con las GPUs de Modal.
        # Qwen3 requiere vLLM >= 0.8.5 (y transformers >= 4.51, que vLLM arrastra).
        "vllm>=0.8.5",
        # `vcot.pipeline` valida las etapas con pydantic.
        "pydantic>=2",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
            # La imagen no trae CUDA toolkit (nvcc), así que evitamos los kernels
            # que se compilan en runtime (JIT) y necesitarían nvcc:
            #  - sampler de FlashInfer → usa el sampler nativo de PyTorch.
            #  - DeepGEMM (kernels FP8) → no se usa con Qwen3 en bf16; solo ruido.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_DEEP_GEMM": "0",
        }
    )
    # Trae el paquete local `vcot` (pipeline + telemetry) al contenedor.
    .add_local_python_source("vcot")
)

app = modal.App("vcot-planner", image=image)

hf_cache = modal.Volume.from_name("vcot-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("vcot-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# --------------------------------------------------------------------------- #
# Cliente vLLM in-process (implementa vcot.pipeline.llm.LLMClient)
# --------------------------------------------------------------------------- #


class _VLLMClient:
    """Adapter del motor vLLM al contrato `LLMClient` del planner.

    Mantiene `model`/`base_url` para que el planner los registre en `meta`.
    """

    def __init__(self, llm, sampling_params, model: str) -> None:
        self._llm = llm
        self._sp = sampling_params
        self.model = model
        self.base_url = None  # in-process: no hay endpoint

    def complete(self, system: str, user: str, *, json_mode: bool = True):
        from vcot.pipeline.llm import LLMResponse

        outs = self._llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self._sp,
            use_tqdm=False,
            # Thinking OFF: en V-CoT el razonamiento explícito son las etapas N1–N6,
            # no un bloque <think>. Da JSON limpio y menos tokens (menos coste).
            chat_template_kwargs={"enable_thinking": False},
        )
        out = outs[0]
        return LLMResponse(
            text=out.outputs[0].text,
            input_tokens=len(out.prompt_token_ids),
            output_tokens=len(out.outputs[0].token_ids),
        )


# --------------------------------------------------------------------------- #
# Planner N1–N6
# --------------------------------------------------------------------------- #


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: hf_cache, OUTPUT_DIR: outputs},
    secrets=[hf_secret],
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW,  # mantener el contenedor ~2 min tras la última llamada
)
class Planner:
    @modal.enter()
    def load(self):
        """Carga el motor vLLM una vez por contenedor (amortiza model_load_s)."""
        from vcot.telemetry import ContainerMeter

        # Medidor de coste real: arranca antes de cargar vLLM (la carga se factura).
        self.meter = ContainerMeter(GPU)

        from vllm import LLM, SamplingParams

        t0 = time.perf_counter()
        self.llm = LLM(
            model=CFG["repo"],
            dtype="bfloat16",
            max_model_len=CFG["max_model_len"],
            gpu_memory_utilization=0.90,
            download_dir=CACHE_DIR,
            enforce_eager=True,
        )
        # Parámetros recomendados por Qwen para Qwen3 en modo NO-thinking
        # (greedy decoding está desaconsejado en Qwen3). Se guardan los kwargs
        # para derivar una copia con `seed` fijo por llamada (traza reproducible).
        self._sampling_kwargs = dict(
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=1024
        )
        self.sampling = SamplingParams(**self._sampling_kwargs)
        self.model_load_s = time.perf_counter() - t0
        print(f"[load] {CFG['repo']} en {GPU} en {self.model_load_s:.1f}s")

    @modal.exit()
    def _bill(self):
        """Coste REAL del contenedor: vida completa (carga + plans + idle tail)."""
        meter = getattr(self, "meter", None)
        if meter is None:  # load() falló antes de crear el medidor
            return
        cost = meter.stop()
        record = {
            "kind": "planner",
            "model": MODEL,
            "model_load_s": round(getattr(self, "model_load_s", 0.0), 3),
            **cost.as_dict(),
        }
        try:
            with open(f"{OUTPUT_DIR}/container_costs.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            outputs.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"[cost] no se pudo persistir container_costs.jsonl: {exc}")
        print(
            f"[cost] contenedor REAL ${cost.real_cost_usd:.5f} · {cost.billed_s:.1f}s vida "
            f"(GPU ${cost.gpu_cost_usd:.5f} + CPU ${cost.cpu_cost_usd:.5f} + "
            f"mem ${cost.mem_cost_usd:.5f}, {cost.mem_gib:.1f} GiB pico)"
        )

    @modal.method()
    def plan(self, prompt: str, seed: int | None = None) -> dict:
        """Genera la traza V-CoT (N1–N6) con coste **real** por etapa.

        Con ``seed`` fijo la traza es reproducible (mismo prompt+seed → misma
        cadena N1–N6); sin él, el sampling es estocástico (temperature=0.7).
        """
        from vllm import SamplingParams

        from vcot.pipeline.planner import Planner as PipelinePlanner

        sampling = (
            SamplingParams(**{**self._sampling_kwargs, "seed": seed})
            if seed is not None
            else self.sampling
        )
        client = _VLLMClient(self.llm, sampling, CFG["repo"])
        # projected_gpu = GPU real ⇒ cost_timer mide el coste real de Modal.
        pipeline = PipelinePlanner(client, projected_gpu=GPU, max_retries=2)
        trace = pipeline.plan(prompt, sample_id=uuid.uuid4().hex)

        record = trace.model_dump()
        record["meta"]["model_load_s"] = round(self.model_load_s, 3)
        record["meta"]["execution"] = "modal"  # el coste de telemetría es real, no proyectado
        record["meta"]["seed"] = seed  # reproducibilidad de la traza

        # Persistir la traza en el Volume (dataset incremental de pensamiento visual).
        with open(f"{OUTPUT_DIR}/{trace.id}.trace.json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
        with open(f"{OUTPUT_DIR}/traces.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs.commit()

        print(
            f"[plan] {trace.id} · {trace.total_compute_s:.2f}s · "
            f"${trace.total_projected_cost_usd:.6f} (GPU {GPU})"
        )
        return record


# --------------------------------------------------------------------------- #
# Entrypoint local
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(
    prompt: str = (
        "a lone astronaut inside an abandoned gothic cathedral, "
        "moonlight through stained glass, volumetric fog, cinematic"
    ),
    seed: int = -1,
    out: str = "outputs",
):
    """Genera una traza en Modal y la guarda en local junto a su telemetría."""
    from vcot.reporting.runlog import track_run
    from vcot.telemetry import projected_container_cost

    os.makedirs(out, exist_ok=True)
    with track_run(os.path.join(out, "runs.jsonl"), kind="planner", model=CFG["repo"], gpu=GPU) as run:
        record = Planner().plan.remote(prompt, seed=None if seed < 0 else seed)
        active_s = sum(t["compute_s"] for t in record["telemetry"].values())
        proj = projected_container_cost(
            gpu=GPU,
            active_s=active_s,
            model_load_s=record["meta"].get("model_load_s", 0.0),
            scaledown_window=SCALEDOWN_WINDOW,
        )
        run["n_items"] = 1
        run["total_cost_usd"] = sum(t["projected_cost_usd"] for t in record["telemetry"].values())
        run["real_cost_est_usd"] = proj.real_cost_usd

    path = os.path.join(out, f"{record['id']}.trace.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    # Acumula al dataset local (además del Volume) para el informe final.
    with open(os.path.join(out, "traces.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nVisual tokens:\n  " + " ".join(record["visual_tokens"]))
    print(f"\nTraza -> {path}")
    print("Telemetría — coste marginal por etapa (solo inferencia):")
    marginal = 0.0
    for stage, tele in record["telemetry"].items():
        marginal += tele["projected_cost_usd"]
        print(
            f"  {stage:<14} {tele['compute_s']:6.2f}s  "
            f"{tele['output_tokens']:5d} tok  ${tele['projected_cost_usd']:.6f}"
        )
    print(
        f"\nCoste marginal (inferencia):  ${marginal:.6f}\n"
        f"Coste REAL estimado (carga + plan + idle {SCALEDOWN_WINDOW}s): "
        f"${proj.real_cost_usd:.6f}"
        + (f"  ({proj.real_cost_usd / marginal:.1f}× el marginal)" if marginal else "")
        + "\n  → cifra exacta en outputs/container_costs.jsonl (la mide @modal.exit)"
    )


@app.local_entrypoint()
def generate(prompts_file: str = "", limit: int = 0, out: str = "outputs"):
    """Genera el dataset en paralelo (fan-out con .map) — IDEA.md §4.3.

    Sin ``--prompts-file`` usa los prompts semilla de ``vcot.dataset``. Cada traza
    se persiste en el Volume ``vcot-outputs`` y se acumula en local
    (``outputs/traces.jsonl``); la ejecución queda en ``outputs/runs.jsonl``.

        modal run modal_app/planner.py::generate --limit 5
        modal run modal_app/planner.py::generate --prompts-file prompts.txt
    """
    from vcot.dataset import SEED_PROMPTS, generate_prompts
    from vcot.reporting.runlog import track_run

    if prompts_file:
        with open(prompts_file, encoding="utf-8") as fh:
            prompts = [ln.strip() for ln in fh if ln.strip()]
        if limit:
            prompts = prompts[:limit]
    elif limit:
        # Expansión estratificada determinista (núcleo curado + generados) para
        # llegar a `limit` aunque supere los prompts curados (p.ej. Smoke 100).
        prompts = generate_prompts(limit)
    else:
        prompts = list(SEED_PROMPTS)

    os.makedirs(out, exist_ok=True)
    traces_path = os.path.join(out, "traces.jsonl")
    print(f"Generando {len(prompts)} trazas en Modal ({GPU}) …")

    n = 0
    total_cost = 0.0
    from modal._utils.async_utils import ExceptionWrapper

    with track_run(
        os.path.join(out, "runs.jsonl"),
        kind="dataset",
        model=CFG["repo"],
        gpu=GPU,
        params={"n_prompts": len(prompts)},
    ) as run:
        with open(traces_path, "a", encoding="utf-8") as fh:
            for index, item in enumerate(
                Planner().plan.map(prompts, return_exceptions=True), start=1
            ):
                if isinstance(item, ExceptionWrapper) or isinstance(item, BaseException):
                    prompt = prompts[index - 1] if index <= len(prompts) else "<unknown>"
                    error_obj = item.value if isinstance(item, ExceptionWrapper) else item
                    print(
                        f"WARNING: prompt #{index} failed while planning: {prompt!r}\n"
                        f"  error: {type(error_obj).__name__}: {error_obj}",
                        file=sys.stderr,
                    )
                    continue

                rec = item
                try:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except TypeError:
                    prompt = prompts[index - 1] if index <= len(prompts) else "<unknown>"
                    print(
                        f"WARNING: prompt #{index} produced a non-serializable result: {prompt!r}\n"
                        f"  result type: {type(rec).__name__} repr: {repr(rec)}",
                        file=sys.stderr,
                    )
                    continue
                n += 1
                total_cost += sum(t["projected_cost_usd"] for t in rec["telemetry"].values())
        run["n_items"] = n
        run["total_cost_usd"] = total_cost

    avg = total_cost / n if n else 0.0
    print(
        f"\n{n} trazas generadas · coste MARGINAL razonamiento ≈ ${total_cost:.4f} "
        f"(${avg:.6f}/traza) · dataset en {traces_path} (+ Volume vcot-outputs)"
    )
    print(
        "Coste REAL (carga + idle de cada contenedor del fan-out): se mide por "
        "contenedor en el Volume → outputs/container_costs.jsonl. Súmalo para el gasto real."
    )
    print(f"Informe final:  python -m vcot.reporting {out}")
