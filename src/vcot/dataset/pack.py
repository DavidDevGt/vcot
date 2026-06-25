"""Empaquetado del dataset V-CoT a shards WebDataset + datacard (IDEA.md §4.2).

Convierte ``traces.eval.jsonl`` (trazas con bloque ``dataset`` ya evaluado) en un
dataset **distribuible y versionado**:

- ``shards/shard-NNNNN.tar`` — shards WebDataset: por muestra, el ``{id}.json``
  (la traza completa) + sus imágenes ``{id}.{idx}.webp`` (clave = ``trace.id``).
- ``index.jsonl`` — una fila por muestra (split, gate, métricas) para filtrar sin
  abrir los tars.
- ``manifest.json`` — versión, conteos, git sha, versiones de modelo, política de
  semillas, splits, umbrales del gate (reproducibilidad).
- ``DATACARD.md`` — *Datasheet for Datasets* autogenerada (lo que un paper exige).

Stdlib pura (``tarfile``/``json``): testeable sin dependencias. El formato sigue la
decisión del proyecto: **JSONL + WebDataset shards**.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

SCHEMA_VERSION = "vcot-dataset/1.0"
DEFAULT_RENDERER = "black-forest-labs/FLUX.2-klein-9B"
EVAL_MODELS = {
    "clip": "open_clip ViT-L-14 (openai)",
    "aesthetic": "LAION improved-aesthetic-predictor (ViT-L/14 linear)",
    "image_reward": "ImageReward-v1.0 (preferencia humana)",
    "faithfulness": "google/owlv2-base-patch16-ensemble (open-vocab detection → IoU)",
    "nsfw": "Falconsai/nsfw_image_detection",
    "dedup": "pHash DCT 64-bit (Hamming)",
}
#: Política de semillas REAL del pipeline (ver vcot.dataset.seedgen.derive_seed).
SEED_POLICY = (
    "seed = derive_seed(prompt) (sha256, determinista) → planner (meta.seed) y "
    "renderer (images[].seed); split por hash(prompt) sin fuga ⇒ regenerable bit-a-bit"
)


def _image_paths(record: Dict, images_dir: str) -> List[str]:
    """Rutas locales de las imágenes de una muestra ({id}_{idx}.webp)."""
    refs = record.get("images") or []
    if refs:
        idxs = [int(r.get("idx", i)) for i, r in enumerate(refs)]
    else:
        n = record.get("meta", {}).get("n_variations") or 0
        idxs = list(range(n))
    paths = []
    for i in idxs:
        p = os.path.join(images_dir, f"{record['id']}_{i}.webp")
        if os.path.exists(p):
            paths.append((i, p))
    return paths


def _quality(record: Dict) -> Dict:
    return (record.get("dataset") or {}).get("quality") or {}


def _safety(record: Dict) -> Dict:
    return (record.get("dataset") or {}).get("safety") or {}


def build_index_row(record: Dict, shard: str, n_images: int) -> Dict:
    ds = record.get("dataset") or {}
    q, s = _quality(record), _safety(record)
    return {
        "id": record["id"],
        "prompt": record["prompt"],
        "stratum": ds.get("stratum"),
        "split": ds.get("split"),
        "seed": record.get("meta", {}).get("seed"),
        "passed_gate": q.get("passed_gate"),
        "gate_reasons": q.get("gate_reasons", []),
        "clip_score": q.get("clip_score"),
        "aesthetic": q.get("aesthetic"),
        "image_reward": q.get("image_reward"),
        "faithfulness": q.get("faithfulness"),
        "detection_coverage": q.get("detection_coverage"),
        "nsfw": s.get("nsfw"),
        "nsfw_label": s.get("nsfw_label"),
        "release_blocked": s.get("release_blocked"),
        "is_duplicate": s.get("is_duplicate"),
        "n_images": n_images,
        "shard": shard,
    }


def write_shards(
    records: Sequence[Dict],
    images_dir: str,
    out_dir: str,
    *,
    shard_size: int = 256,
) -> tuple[List[str], int, List[Dict]]:
    """Escribe los shards .tar y devuelve ``(nombres, n_imágenes, filas_index)``."""
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    shard_names: List[str] = []
    index_rows: List[Dict] = []
    n_images_total = 0
    tar: Optional[tarfile.TarFile] = None
    shard_name = ""

    for i, record in enumerate(records):
        if i % shard_size == 0:
            if tar is not None:
                tar.close()
            shard_name = f"shard-{len(shard_names):05d}.tar"
            shard_names.append(shard_name)
            tar = tarfile.open(os.path.join(shards_dir, shard_name), "w")

        key = record["id"]
        # Traza completa como sidecar JSON (clave = trace.id).
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
        _add_bytes(tar, f"{key}.json", payload)

        imgs = _image_paths(record, images_dir)
        for idx, path in imgs:
            tar.add(path, arcname=f"{key}.{idx}.webp")
        n_images_total += len(imgs)
        index_rows.append(build_index_row(record, shard_name, len(imgs)))

    if tar is not None:
        tar.close()
    return shard_names, n_images_total, index_rows


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    import io

    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def summarize(records: Sequence[Dict]) -> Dict:
    """Estadísticos para el manifiesto/datacard (splits, gate, métricas medias)."""
    splits = {"train": 0, "val": 0, "test": 0, None: 0}
    gate = {"passed": 0, "failed": 0, "unknown": 0}
    metrics: Dict[str, List[float]] = {
        "clip_score": [], "aesthetic": [], "image_reward": [],
        "faithfulness": [], "detection_coverage": [],
    }
    strata: Dict[str, int] = {}
    n_dup = 0
    n_blocked = 0
    for r in records:
        ds = r.get("dataset") or {}
        splits[ds.get("split")] = splits.get(ds.get("split"), 0) + 1
        strata[ds.get("stratum") or "unknown"] = strata.get(ds.get("stratum") or "unknown", 0) + 1
        q, s = _quality(r), _safety(r)
        passed = q.get("passed_gate")
        gate["passed" if passed else "failed" if passed is False else "unknown"] += 1
        for k in metrics:
            if q.get(k) is not None:
                metrics[k].append(q[k])
        if s.get("is_duplicate"):
            n_dup += 1
        if s.get("release_blocked"):
            n_blocked += 1
    means = {k: (round(sum(v) / len(v), 4) if v else None) for k, v in metrics.items()}
    return {
        "splits": {k or "unassigned": n for k, n in splits.items() if n or k},
        "strata": dict(sorted(strata.items())),
        "gate": gate,
        "means": means,
        "n_duplicates": n_dup,
        "n_release_blocked": n_blocked,
    }


def build_manifest(
    records: Sequence[Dict],
    shard_names: Sequence[str],
    n_images: int,
    stats: Dict,
    *,
    name: str,
    version: str,
    renderer: str,
) -> Dict:
    first = records[0] if records else {}
    ds0 = first.get("dataset") or {}
    return {
        "schema": SCHEMA_VERSION,
        "name": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": ds0.get("code_version"),
        "license": ds0.get("license"),
        "n_samples": len(records),
        "n_images": n_images,
        "splits": stats["splits"],
        "strata": stats["strata"],
        "gate": stats["gate"],
        "metric_means": stats["means"],
        "n_duplicates": stats["n_duplicates"],
        "n_release_blocked": stats["n_release_blocked"],
        "models": {
            "planner": first.get("meta", {}).get("planner"),
            "renderer": renderer,
            "eval": EVAL_MODELS,
        },
        "seed_policy": SEED_POLICY,
        "shards": list(shard_names),
        "format": "WebDataset (.tar): {id}.json + {id}.{idx}.webp por muestra",
    }


def render_datacard(manifest: Dict) -> str:
    """Datasheet for Datasets (Gebru et al.) a partir del manifiesto."""
    m = manifest
    splits = ", ".join(f"{k}={v}" for k, v in m["splits"].items())
    means = m["metric_means"]
    return f"""# Datacard — {m['name']} ({m['version']})

> Generado automáticamente · {m['created_at']} · esquema `{m['schema']}`
> código `{m.get('code_version')}` · licencia: {m.get('license')}

## Motivation
Dataset de **pensamiento visual** (Visual Chain-of-Thought): captura la *cadena de
decisiones* N1–N6 (intención→layout→composición→iluminación→materiales→color) que
precede a cada imagen, más las imágenes renderizadas y su evaluación. No es un
dataset de imágenes, sino del **proceso** que las produce (IDEA.md §1).

## Composition
- **{m['n_samples']} muestras**, **{m['n_images']} imágenes** ({m['models']['renderer']}).
- Cada muestra: traza V-CoT validada (pydantic) + imágenes ligadas por `trace.id`
  (`sha256` por variación) + bloque `dataset` (quality/safety/split).
- Splits (sin fuga, por hash del prompt): {splits}.
- Gate de calidad: {m['gate']['passed']} pasan / {m['gate']['failed']} no / {m['gate']['unknown']} sin evaluar.
- Duplicados perceptuales marcados: {m['n_duplicates']}.

## Collection process
- **Planner**: {m['models']['planner']} (open-weights, vLLM en Modal serverless GPU),
  razonamiento N1–N6 con salida estructurada validada por etapa.
- **Renderer**: {m['models']['renderer']} (N7, 4 variaciones/batch).
- Prompts: briefs creativos sub-especificados, estratificados por género/escala/época
  (`vcot.dataset.generate_prompts`, determinista por semilla → reproducible).
- **Semillas**: `{m.get('seed_policy')}`.

## Preprocessing / cleaning / labeling
- **CLIP score** (alineación prompt↔imagen): {EVAL_MODELS['clip']} — media {means['clip_score']}.
- **ImageReward** (preferencia humana, triangula CLIP): media {means.get('image_reward')}.
- **Aesthetic**: {EVAL_MODELS['aesthetic']} — media {means['aesthetic']}.
- **Layout-faithfulness** (IoU entidades N2 ↔ detección): media {means['faithfulness']}
  con **detection_coverage** {means.get('detection_coverage')} (separa "no detectado"
  de "mal ubicado"). ⚠️ Es una **línea base**: el render usa solo prompt-enrichment,
  el layout NO condiciona el píxel → mide el *gap*, no un mecanismo de control.
- **NSFW**: {EVAL_MODELS['nsfw']}. **Dedup**: {EVAL_MODELS['dedup']}.
- Gate de calidad con umbrales versionados (calibrables con `vcot.eval.calibration`
  contra juicio humano); cada muestra lleva `passed_gate` + razones.

## Uses
Destilación de un modelo pequeño que *razona* visualmente (IDEA.md §5); estudio de
**control vs. fidelidad** (faithfulness como gap baseline); investigación de costes
de inferencia serverless.

## Safety
- NSFW por muestra con etiqueta `ok`/`review`/`blocked`; **{m.get('n_release_blocked', 0)}**
  marcadas `release_blocked`.
- ⚠️ **Antes de cualquier release público faltan** `csam_hash` (match contra bases
  de hashes conocidas) y `pii_scan` (rostros identificables / PII) — declarados como
  pendientes en `vcot.eval.safety`. Este dataset **no está cleared para release abierto**.

## Distribution
Shards WebDataset (`shards/*.tar`) + `index.jsonl` (filtrado por split/gate/stratum sin
abrir los tars) + este datacard. **Licencia: {m.get('license')}** — respetá la licencia
non-commercial de FLUX.2 al redistribuir las imágenes.

## Maintenance
Versionado por `manifest.json` (`version`, `code_version` git sha, modelos+revisiones,
`created_at`). **Regenerable bit-a-bit** desde los prompts semilla por la política de
semillas determinista.
"""


def pack(
    traces_path: str,
    images_dir: str,
    out_dir: str,
    *,
    name: str = "vcot-dataset",
    version: str = "0.1.0",
    shard_size: int = 256,
    renderer: str = DEFAULT_RENDERER,
    only_passed: bool = False,
) -> Dict:
    """Empaqueta el dataset y devuelve el manifiesto (escrito en ``out_dir``)."""
    with open(traces_path, encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()]
    if only_passed:
        records = [r for r in records if _quality(r).get("passed_gate")]

    os.makedirs(out_dir, exist_ok=True)
    shard_names, n_images, index_rows = write_shards(
        records, images_dir, out_dir, shard_size=shard_size
    )
    stats = summarize(records)
    manifest = build_manifest(
        records, shard_names, n_images, stats,
        name=name, version=version, renderer=renderer,
    )

    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as fh:
        for row in index_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "DATACARD.md"), "w", encoding="utf-8") as fh:
        fh.write(render_datacard(manifest))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-pack",
        description="Empaqueta el dataset V-CoT en shards WebDataset + datacard.",
    )
    parser.add_argument("traces", help="Ruta a traces.eval.jsonl (o traces.jsonl).")
    parser.add_argument("--images", default="outputs", help="Carpeta con las imágenes.")
    parser.add_argument("--out", default="dataset", help="Carpeta de salida del dataset.")
    parser.add_argument("--name", default="vcot-dataset")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--only-passed", action="store_true", help="Solo muestras que pasan el gate.")
    args = parser.parse_args(argv)

    manifest = pack(
        args.traces, args.images, args.out,
        name=args.name, version=args.version,
        shard_size=args.shard_size, only_passed=args.only_passed,
    )
    print(
        f"{manifest['n_samples']} muestras · {manifest['n_images']} imágenes · "
        f"{len(manifest['shards'])} shards -> {args.out}/"
    )
    print(f"  index.jsonl · manifest.json · DATACARD.md")


if __name__ == "__main__":
    main()
