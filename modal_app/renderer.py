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
        "git+https://github.com/huggingface/diffusers.git",
        "huggingface-hub==0.36.0",
        "safetensors==0.5.3",
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
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        height: int = 1024,
        width: int = 1024,
    ) -> dict:
        """Genera una imagen y devuelve telemetría + bytes WebP (IDEA.md §4.2, §6)."""
        import torch
        from vcot.telemetry import cost_timer  # noqa: WPS433 (import diferido al contenedor)

        steps = steps or CFG["steps"]
        guidance = CFG["guidance"] if guidance is None else guidance
        generator = (
            torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
        )

        with cost_timer(gpu=GPU) as t:
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                generator=generator,
            )
        pil_image = result.images[0]

        buf = io.BytesIO()
        pil_image.save(buf, format="WEBP", quality=92)
        image_bytes = buf.getvalue()

        sample_id = uuid.uuid4().hex
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
            },
            "telemetry": {
                "render": t.as_dict(),  # {compute_s, rate_usd_per_s, cost_usd}
                "model_load_s": round(self.model_load_s, 3),
            },
            "final_image": f"{OUTPUT_DIR}/{sample_id}.webp",
        }

        # Persistir imagen + registro JSONL en el Volume (dataset incremental).
        with open(f"{OUTPUT_DIR}/{sample_id}.webp", "wb") as fh:
            fh.write(image_bytes)
        with open(f"{OUTPUT_DIR}/records.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
        outputs.commit()

        print(
            f"[render] {sample_id} · {t.seconds:.2f}s · ${t.cost:.5f} "
            f"({steps} pasos, {GPU})"
        )
        record["_image_bytes"] = image_bytes  # se quita antes de imprimir en local
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
    steps: int = 0,
    seed: int = -1,
    out: str = "outputs",
):
    """Genera una imagen y la guarda en local junto a su telemetría."""
    record = Renderer().render.remote(
        prompt,
        steps=steps or None,
        seed=None if seed < 0 else seed,
    )
    image_bytes = record.pop("_image_bytes")

    os.makedirs(out, exist_ok=True)
    img_path = os.path.join(out, f"{record['id']}.webp")
    with open(img_path, "wb") as fh:
        fh.write(image_bytes)

    print(f"\nImagen  -> {img_path}")
    print("Telemetría:")
    print(json.dumps(record["telemetry"], indent=2, ensure_ascii=False))
