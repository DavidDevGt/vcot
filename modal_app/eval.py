"""Eval research-grade del dataset sobre Modal GPU (IDEA.md §4, §8).

Recorre el dataset generado (``outputs/traces.jsonl`` + imágenes locales) y
puntúa cada muestra con modelos open-weights, **apoyándose en el núcleo puro de
``vcot.eval``** para la matemática (faithfulness/dedup/splits/gate):

- **CLIP score** (alineación prompt↔imagen, open_clip ViT-L/14),
- **aesthetic** (predictor lineal LAION sobre el embedding CLIP),
- **faithfulness** (OWLv2 detecta las entidades → IoU vs las bboxes de N2),
- **NSFW** (clasificador de seguridad),
- **pHash** para dedup perceptual a nivel de dataset.

Cada scorer está **guardado**: si un modelo no carga, su score queda ``None`` y
no tumba la corrida (degradación elegante). Escribe ``outputs/traces.eval.jsonl``
con el bloque ``dataset`` (quality/safety/split) listo para empaquetar (Fase 4).

    modal run modal_app/eval.py::evaluate                       # sobre outputs/traces.jsonl
    modal run modal_app/eval.py::evaluate --traces otro.jsonl --images outputs
"""

from __future__ import annotations

import json
import os
import sys

import modal

GPU = os.environ.get("VCOT_EVAL_GPU", "L4")  # 24 GB: cabe CLIP-L + OWLv2-base + NSFW
CACHE_DIR = "/cache"
OUTPUT_DIR = "/outputs"
OWL_THRESHOLD = float(os.environ.get("VCOT_OWL_THRESHOLD", "0.1"))

#: Segundos que Modal factura el contenedor tras la última llamada (idle tail).
SCALEDOWN_WINDOW = 120

#: **Pins de reproducibilidad.** Para un dataset reproducible cada modelo se fija
#: a un commit (``revision``) y los pesos sueltos se verifican por hash. Acá está
#: el *mecanismo*: poné el commit/sha real antes de un release. ``None`` = usa la
#: rama por defecto y solo **registra** el repo (sin verificación dura).
MODEL_REVISIONS = {
    "clip": ("ViT-L-14", "openai"),         # open_clip: pretrained tag estable
    "owlv2": ("google/owlv2-base-patch16-ensemble", None),
    "nsfw": ("Falconsai/nsfw_image_detection", None),
    "image_reward": ("ImageReward-v1.0", None),
}

#: Pesos del predictor estético LAION (entrenado sobre embeddings ViT-L/14 OpenAI).
AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/"
    "raw/main/sac+logos+ava1-l14-linearMSE.pth"
)
#: sha256 esperado de los pesos aesthetic. ``None`` = se registra el observado sin
#: verificar; poné el valor real para activar la verificación dura.
AESTHETIC_SHA256 = None

#: Provenance que se escribe en el resumen del eval (auditoría/manifiesto).
MODEL_PROVENANCE = {
    "clip": "open_clip ViT-L-14 (openai)",
    "aesthetic": {"url": AESTHETIC_URL, "sha256_expected": AESTHETIC_SHA256},
    "faithfulness": MODEL_REVISIONS["owlv2"][0],
    "image_reward": MODEL_REVISIONS["image_reward"][0],
    "nsfw": MODEL_REVISIONS["nsfw"][0],
    "owl_threshold": OWL_THRESHOLD,
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.7.1",
        "open_clip_torch",
        "transformers>=4.57.0",
        "image-reward",
        "Pillow~=11.2.1",
        "ftfy",
        "regex",
        "pydantic>=2",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"HF_HOME": CACHE_DIR})
    .add_local_python_source("vcot")
)

app = modal.App("vcot-eval", image=image)
hf_cache = modal.Volume.from_name("vcot-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("vcot-outputs", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: hf_cache, OUTPUT_DIR: outputs},
    secrets=[hf_secret],
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW,
)
class Scorer:
    @modal.enter()
    def load(self):
        from vcot.telemetry import ContainerMeter

        # Medidor de coste real: arranca antes de cargar los scorers (la carga
        # de CLIP/OWLv2/NSFW también se factura toda la vida del contenedor).
        self.meter = ContainerMeter(GPU)

        import torch

        self.torch = torch
        self.device = "cuda"
        self.owl_threshold = OWL_THRESHOLD

        # — CLIP ViT-L/14 (clip score + embedding para aesthetic) —
        import open_clip

        arch, pretrained = MODEL_REVISIONS["clip"]
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, cache_dir=CACHE_DIR
        )
        self.clip = self.clip.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

        # — Aesthetic head (guardado + verificación de hash de pesos) —
        self.aesthetic = None
        try:
            import hashlib

            head = torch.nn.Linear(768, 1)
            weights_path = os.path.join(CACHE_DIR, "aesthetic_l14.pth")
            if not os.path.exists(weights_path):
                torch.hub.download_url_to_file(AESTHETIC_URL, weights_path)
            with open(weights_path, "rb") as wf:
                digest = hashlib.sha256(wf.read()).hexdigest()
            if AESTHETIC_SHA256 and digest != AESTHETIC_SHA256:
                raise RuntimeError(f"sha256 aesthetic {digest} != esperado {AESTHETIC_SHA256}")
            self.aesthetic_sha256 = digest  # registrado en provenance
            head.load_state_dict(torch.load(weights_path, map_location="cpu"))
            self.aesthetic = head.to(self.device).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] aesthetic head no disponible ({exc}); score=None")

        # — OWLv2 (detección open-vocab → faithfulness) —
        self.owl = self.owl_proc = None
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            repo, rev = MODEL_REVISIONS["owlv2"]
            self.owl_proc = Owlv2Processor.from_pretrained(repo, revision=rev, cache_dir=CACHE_DIR)
            self.owl = (
                Owlv2ForObjectDetection.from_pretrained(repo, revision=rev, cache_dir=CACHE_DIR)
                .to(self.device)
                .eval()
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] OWLv2 no disponible ({exc}); faithfulness=None")

        # — NSFW (seguridad) —
        self.nsfw = None
        try:
            from transformers import pipeline

            repo, rev = MODEL_REVISIONS["nsfw"]
            self.nsfw = pipeline("image-classification", model=repo, revision=rev, device=0)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] NSFW no disponible ({exc}); nsfw=None")

        # — ImageReward (triangulación de alineación/preferencia humana) —
        self.image_reward = None
        try:
            import ImageReward as RM

            self.image_reward = RM.load(MODEL_REVISIONS["image_reward"][0], download_root=CACHE_DIR)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] ImageReward no disponible ({exc}); image_reward=None")

        # Refrescar el Volume para ver las imágenes recién commiteadas por el render.
        try:
            outputs.reload()
        except Exception:  # noqa: BLE001
            pass

    @modal.exit()
    def _bill(self):
        """Coste REAL del contenedor de eval: vida completa (carga + scores + idle)."""
        meter = getattr(self, "meter", None)
        if meter is None:  # load() falló antes de crear el medidor
            return
        cost = meter.stop()
        record = {"kind": "eval", **cost.as_dict()}
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

    # ----------------------------------------------------------------------- #

    def _clip_and_aesthetic(self, pil_img, prompt: str):
        torch = self.torch
        with torch.no_grad():
            px = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            img_feat = self.clip.encode_image(px)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt = self.tokenizer([prompt]).to(self.device)
            txt_feat = self.clip.encode_text(txt)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            clip_score = float((img_feat @ txt_feat.T).item())

            aesthetic = None
            if self.aesthetic is not None:
                aesthetic = float(self.aesthetic(img_feat.float()).item())
        return clip_score, aesthetic

    def _detections(self, pil_img, labels):
        """OWLv2 → lista de detecciones ``{label, bbox(norm), score}``.

        Consulta el detector con texto natural (``detector_query``: snake_case →
        palabras) pero conserva el id de entidad original como ``label`` para que
        el emparejamiento de faithfulness alinee. Las imágenes son cuadradas
        (1024²) ⇒ el padding-a-cuadrado de OWLv2 no distorsiona las cajas.
        """
        from vcot.eval.faithfulness import detector_query

        if self.owl is None or not labels:
            return []
        torch = self.torch
        w, h = pil_img.size
        queries_text = [detector_query(l) for l in labels]
        with torch.no_grad():
            inputs = self.owl_proc(text=[queries_text], images=pil_img, return_tensors="pt").to(
                self.device
            )
            outputs = self.owl(**inputs)
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.owl_proc.post_process_object_detection(
                outputs, threshold=self.owl_threshold, target_sizes=target_sizes
            )[0]
        dets = []
        for box, score, label_idx in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            dets.append(
                {
                    "label": labels[int(label_idx)],  # id de entidad original (no la query)
                    "bbox": (x0 / w, y0 / h, x1 / w, y1 / h),
                    "score": float(score),
                }
            )
        return dets

    def _nsfw_prob(self, pil_img):
        if self.nsfw is None:
            return None
        preds = self.nsfw(pil_img)
        for p in preds:
            if str(p["label"]).lower() == "nsfw":
                return float(p["score"])
        return 0.0

    def _image_reward(self, pil_img, prompt: str):
        """Score de preferencia humana (ImageReward); triangula con CLIP/aesthetic."""
        if self.image_reward is None:
            return None
        try:
            return round(float(self.image_reward.score(prompt, [pil_img])), 4)
        except Exception:  # noqa: BLE001
            return None

    @modal.method()
    def score(self, payload: dict) -> dict:
        """Puntúa las N variaciones. ``payload`` = {prompt, entities, image_paths}.

        Lee las imágenes del **Volume montado** (``/outputs/...``), no de bytes
        enviados por la red — así no se duplica el tráfico ni se carga el cliente.
        """
        from PIL import Image

        from vcot.eval import layout_faithfulness, phash_dct

        prompt = payload["prompt"]
        entities = payload["entities"]
        labels = [e["label"] for e in entities]
        image_paths = payload["image_paths"]

        per_image = []
        primary_phash = None
        for idx, path in enumerate(image_paths):
            pil = Image.open(path).convert("RGB")
            clip_score, aesthetic = self._clip_and_aesthetic(pil, prompt)
            dets = self._detections(pil, labels)
            faith = layout_faithfulness(entities, dets)
            nsfw = self._nsfw_prob(pil)
            phash = phash_dct(pil)
            if idx == 0:
                primary_phash = phash
            per_image.append(
                {
                    "idx": idx, "clip_score": round(clip_score, 4),
                    "aesthetic": round(aesthetic, 3) if aesthetic is not None else None,
                    "image_reward": self._image_reward(pil, prompt),
                    "faithfulness": faith["score"],
                    "detection_coverage": faith["detection_coverage"],
                    "nsfw": round(nsfw, 4) if nsfw is not None else None,
                    "phash": phash,
                }
            )

        return {
            "per_image": per_image,
            "primary_phash": primary_phash,
            "clip_score": _agg(per_image, "clip_score", "mean"),
            "aesthetic": _agg(per_image, "aesthetic", "mean"),
            "image_reward": _agg(per_image, "image_reward", "mean"),
            "faithfulness": _agg(per_image, "faithfulness", "mean"),
            "detection_coverage": _agg(per_image, "detection_coverage", "mean"),
            "nsfw": _agg(per_image, "nsfw", "max"),
        }


def _agg(per_image, key, how):
    vals = [p[key] for p in per_image if p.get(key) is not None]
    if not vals:
        return None
    return round(max(vals) if how == "max" else sum(vals) / len(vals), 4)


@app.local_entrypoint()
def evaluate(traces: str = "outputs/dataset.jsonl", images: str = "outputs", out: str = ""):
    """Evalúa el dataset y escribe ``traces.eval.jsonl`` con quality/safety/split."""
    from vcot.eval import assign_splits, duplicate_indices, merge_eval
    from vcot.eval.faithfulness import detectable_entities
    from vcot.eval.safety import classify_safety
    from vcot.eval.stats import format_quality_report, quality_report
    from vcot.reporting.runlog import track_run

    out = out or (traces[:-6] + ".eval.jsonl" if traces.endswith(".jsonl") else traces + ".eval")
    with open(traces, encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()]
    if not records:
        raise SystemExit(f"sin trazas en {traces}")

    # Rutas de imagen en el Volume montado (las escribió el renderer en /outputs).
    payloads, owners = [], []
    for rec in records:
        ents = detectable_entities(rec.get("layout", {}))
        paths = [im["path"] for im in rec.get("images", []) if im.get("path")]
        if not paths:
            print(f"WARNING: sin imágenes para {rec['id']} (¿corriste generate_full?)",
                  file=sys.stderr)
            continue
        payloads.append({"prompt": rec["prompt"], "entities": ents, "image_paths": paths})
        owners.append(rec)

    print(f"Evaluando {len(payloads)} muestras en Modal ({GPU}) …")
    splits = assign_splits([r["prompt"] for r in owners])

    with track_run(
        os.path.join(images, "runs.jsonl"), kind="eval", gpu=GPU,
        params={"n": len(payloads)},
    ) as run:
        results = list(Scorer().score.map(payloads, return_exceptions=True))
        # Dedup perceptual a nivel dataset por el pHash de la variación principal.
        phashes = [
            r.get("primary_phash", 0) if not isinstance(r, Exception) else 0 for r in results
        ]
        dups = duplicate_indices([h or 0 for h in phashes])

        n_pass = 0
        evaluated: list = []
        with open(out, "w", encoding="utf-8") as ofh:
            for i, (rec, res) in enumerate(zip(owners, results)):
                if isinstance(res, Exception):
                    print(f"WARNING: eval falló para {rec['id']}: {res}", file=sys.stderr)
                    quality, safety = {}, {}
                else:
                    quality = {
                        "clip_score": res["clip_score"], "aesthetic": res["aesthetic"],
                        "image_reward": res["image_reward"],
                        "faithfulness": res["faithfulness"],
                        "detection_coverage": res["detection_coverage"],
                        "per_image": res["per_image"],
                    }
                    safety = classify_safety(res["nsfw"])
                merge_eval(
                    rec, quality=quality, safety=safety,
                    split=splits[rec["prompt"]], is_duplicate=i in dups,
                )
                n_pass += 1 if rec["dataset"]["quality"].get("passed_gate") else 0
                evaluated.append(rec)
                ofh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        run["n_items"] = len(owners)

    # Resumen estadístico research-grade (distribuciones + breakdown por estrato).
    from vcot.eval import BASELINE_NOTE

    report = quality_report(evaluated)
    report["provenance"] = MODEL_PROVENANCE  # modelos/umbrales usados (auditoría)
    report["faithfulness_note"] = BASELINE_NOTE
    report_path = os.path.join(images, "quality.json")
    with open(report_path, "w", encoding="utf-8") as rfh:
        json.dump(report, rfh, indent=2, ensure_ascii=False)

    print(f"\n{len(owners)} evaluadas · {n_pass} pasan el gate · {len(dups)} duplicados\n")
    print(format_quality_report(report))
    print(f"\nDataset evaluado -> {out}  ·  resumen -> {report_path}")
    print(f"Empaquetar:  python -m vcot.dataset.pack {out} --images {images} --out dataset/")
