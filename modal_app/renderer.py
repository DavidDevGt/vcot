"""N7 — Final Render sobre Modal.com (IDEA.md §2, §7).

Renderer de imágenes con FLUX.2, instrumentado con el `cost_timer` de vcot para
medir `compute_s` y `cost_usd` reales por imagen.

Por defecto usa **FLUX.2-klein-9B** (step-distilled, 4 pasos, ~29 GB VRAM) — el
modelo "Klein" que IDEA.md propone destilar. Para cambiar a FLUX.2-dev (32B):

    VCOT_MODEL=dev modal run modal_app/renderer.py --prompt "..."

Requisitos previos (ver modal_app/README.md):
  1. `pip install modal && modal setup`            (autenticación Modal)
  2. Aceptar la licencia de FLUX.2 en HuggingFace   (modelo gated)
  3. Crear un Secret de Modal `huggingface-secret` con HF_TOKEN

Uso:
    modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral"
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid

import modal

# --------------------------------------------------------------------------- #
# Configuración de modelo
# --------------------------------------------------------------------------- #

MODELS = {
    # Klein: step-distilled, 4 pasos, cabe en una sola GPU de 40-80 GB.
    "klein": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline": "Flux2KleinPipeline",
        "steps": 4,
        "guidance": 1.0,
        "gpu": "A100-80GB",
        "offload": False,
    },
    # Dev: 32B, máxima calidad. Necesita mucha VRAM → B200 + offload.
    "dev": {
        "repo": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "steps": 28,
        "guidance": 4.0,
        "gpu": "B200",
        "offload": True,
    },
}

MODEL = os.environ.get("VCOT_MODEL", "klein")
if MODEL not in MODELS:
    raise SystemExit(f"VCOT_MODEL={MODEL!r} no válido. Opciones: {', '.join(MODELS)}")
CFG = MODELS[MODEL]
GPU = os.environ.get("VCOT_GPU", CFG["gpu"])  # debe coincidir con una clave de rates.py

CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"

#: Negativo sugerido (artefactos comunes). Vacío por defecto: el negativo es
#: opt-in porque en FLUX requiere CFG real (~2× coste). Pasalo con --negative.
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, extra limbs, bad anatomy, "
    "watermark, text, signature, jpeg artifacts, oversaturated"
)

# --------------------------------------------------------------------------- #
# Imagen del contenedor (patrón del ejemplo oficial image_to_image de Modal)
# --------------------------------------------------------------------------- #

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "Pillow~=11.2.1",
        "accelerate~=1.8.1",
        # FLUX.2 (Flux2Pipeline / Flux2KleinPipeline) requiere diffusers reciente.
        # No pinear safetensors/huggingface-hub: diffusers-git exige safetensors
        # >=0.8.0 y mueve rápido; dejamos que el resolver elija versiones compatibles.
        "git+https://github.com/huggingface/diffusers.git",
        "huggingface-hub",
        "sentencepiece==0.2.0",
        # transformers reciente para el text-encoder Qwen3 de FLUX.2.
        "transformers>=4.57.0",
        "torch==2.7.1",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    # Trae el paquete local `vcot` (cost_timer + rates) al contenedor.
    .add_local_python_source("vcot")
)

app = modal.App("vcot-renderer", image=image)

hf_cache = modal.Volume.from_name("vcot-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("vcot-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# --------------------------------------------------------------------------- #
# Renderer N7
# --------------------------------------------------------------------------- #


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: hf_cache, OUTPUT_DIR: outputs},
    secrets=[hf_secret],
    timeout=60 * 60,
    scaledown_window=120,  # mantener el contenedor ~2 min tras la última llamada
)
class Renderer:
    @modal.enter()
    def load(self):
        """Carga los pesos una vez por contenedor (amortiza model_load_s)."""
        import torch
        import diffusers

        pipeline_cls = getattr(diffusers, CFG["pipeline"])
        t0 = time.perf_counter()
        self.pipe = pipeline_cls.from_pretrained(
            CFG["repo"],
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        if CFG["offload"]:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cuda")
        self.model_load_s = time.perf_counter() - t0
        print(f"[load] {CFG['repo']} en {GPU} en {self.model_load_s:.1f}s")

    @modal.method()
    def render(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
        n_variations: int = 4,
        true_cfg_scale: float = 1.0,
    ) -> dict:
        """Genera **N variaciones** del mismo prompt; telemetría + bytes WebP (§4.2, §6).

        Las variaciones se generan en un solo batch (`num_images_per_prompt`):
        comparten el text-encoding y el contenedor cálido, así que salen mucho más
        baratas que N llamadas separadas.

        **Prompt negativo:** FLUX es *guidance-distilled*, así que `negative_prompt`
        solo surte efecto con CFG real (`true_cfg_scale > 1`), que ~duplica el
        cómputo. Si das un negativo sin CFG, se sube `true_cfg_scale` a 4.0. Se pasa
        solo si el pipeline lo soporta (FLUX.2 varía según versión de diffusers).
        """
        import inspect
        import torch
        from vcot.telemetry import cost_timer  # noqa: WPS433 (import diferido al contenedor)

        steps = steps or CFG["steps"]
        guidance = CFG["guidance"] if guidance is None else guidance
        n_variations = max(1, n_variations)
        negative_prompt = (negative_prompt or "").strip()
        if negative_prompt and true_cfg_scale <= 1.0:
            true_cfg_scale = 4.0  # sin CFG real el negativo se ignora
        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )

        call_kwargs = dict(
            prompt=prompt,
            num_images_per_prompt=n_variations,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        )
        # Solo pasamos negativo/CFG si el pipeline los acepta (evita TypeError).
        supported = inspect.signature(self.pipe.__call__).parameters
        applied_negative = False
        if negative_prompt and "negative_prompt" in supported:
            call_kwargs["negative_prompt"] = negative_prompt
            if "true_cfg_scale" in supported:
                call_kwargs["true_cfg_scale"] = true_cfg_scale
            applied_negative = True
        elif negative_prompt:
            print(f"[warn] {CFG['pipeline']} no soporta negative_prompt — se ignora")

        with cost_timer(gpu=GPU) as t:
            result = self.pipe(**call_kwargs)

        image_bytes_list = []
        for img in result.images:
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=92)
            image_bytes_list.append(buf.getvalue())
        n = len(image_bytes_list)

        sample_id = uuid.uuid4().hex
        final_images = [f"{OUTPUT_DIR}/{sample_id}_{i}.webp" for i in range(n)]
        render_tel = t.as_dict()  # {compute_s, rate_usd_per_s, cost_usd}
        render_tel["n_variations"] = n
        render_tel["cost_per_image_usd"] = round(t.cost / n, 8) if n else 0.0

        record = {
            "id": sample_id,
            "prompt": prompt,
            "stage": "N7",
            "meta": {
                "model": MODEL,
                "repo": CFG["repo"],
                "gpu": GPU,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
                "resolution": [width, height],
                "n_variations": n,
                "negative_prompt": negative_prompt if applied_negative else "",
                "true_cfg_scale": true_cfg_scale if applied_negative else 1.0,
            },
            "telemetry": {
                "render": render_tel,
                "model_load_s": round(self.model_load_s, 3),
            },
            "final_image": final_images[0],  # variación principal
            "final_images": final_images,
        }

        # Persistir las N variaciones + registro JSONL en el Volume.
        for path, image_bytes in zip(final_images, image_bytes_list):
            with open(path, "wb") as fh:
                fh.write(image_bytes)
        with open(f"{OUTPUT_DIR}/records.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        outputs.commit()

        print(
            f"[render] {sample_id} · {n} variaciones · {t.seconds:.2f}s · ${t.cost:.5f} "
            f"(${render_tel['cost_per_image_usd']:.5f}/img, {steps} pasos, {GPU})"
        )
        record["_image_bytes_list"] = image_bytes_list  # se quita antes de imprimir
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
    negative: str = "",
    steps: int = 0,
    seed: int = -1,
    variations: int = 4,
    cfg: float = 0.0,
    out: str = "outputs",
):
    """Genera N variaciones (4 por defecto) del prompt y las guarda en local.

    `--negative` activa prompt negativo (CFG real, ~2× coste; usá `--negative
    default` para el negativo sugerido). `--cfg` fija true_cfg_scale (0 = auto).
    """
    from vcot.reporting.runlog import track_run

    if negative.strip().lower() == "default":
        negative = DEFAULT_NEGATIVE

    os.makedirs(out, exist_ok=True)
    with track_run(os.path.join(out, "runs.jsonl"), kind="renderer", model=CFG["repo"], gpu=GPU) as run:
        record = Renderer().render.remote(
            prompt,
            negative_prompt=negative,
            steps=steps or None,
            seed=None if seed < 0 else seed,
            n_variations=variations,
            true_cfg_scale=cfg if cfg > 0 else 1.0,
        )
        run["n_items"] = record["meta"]["n_variations"]
        run["total_cost_usd"] = record["telemetry"]["render"]["cost_usd"]
    image_bytes_list = record.pop("_image_bytes_list")

    paths = []
    for i, image_bytes in enumerate(image_bytes_list):
        p = os.path.join(out, f"{record['id']}_{i}.webp")
        with open(p, "wb") as fh:
            fh.write(image_bytes)
        paths.append(p)

    print(f"\n{len(paths)} variaciones ->")
    for p in paths:
        print("  " + p)
    print("Telemetría:")
    print(json.dumps(record["telemetry"], indent=2, ensure_ascii=False))
