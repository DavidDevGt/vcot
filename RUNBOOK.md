# RUNBOOK — Ejecutar V-CoT de punta a punta

Guía operativa: de cero a un **informe final** con todo registrado. Todo corre en
**Modal serverless GPU** (incluido el LLM del razonamiento, Qwen3-8B con vLLM).

> Requisitos (una vez): cuenta en modal.com · cuenta HuggingFace + token *read* ·
> licencia de FLUX.2 aceptada · `modal setup` hecho · Secret `huggingface-secret`
> creado. Si te falta algo, mirá la sección "Setup" al final.

---

## Flujo completo (en orden)

### 0. Preparar la terminal (cada vez que abrís PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
modal secret list          # debe aparecer "huggingface-secret"
```

### 1. Smoke test — razonamiento (Qwen3-8B, N1–N6)

```powershell
modal run modal_app/planner.py::main --prompt "a lone astronaut in a gothic cathedral, moonlight"
```
- ⚠️ La **1ª vez descarga Qwen3-8B (~16 GB)** → lento y facturado (el contenedor está vivo). Luego queda cacheado.
- Guarda la traza en `outputs/`, la acumula en `outputs/traces.jsonl` y registra la corrida en `outputs/runs.jsonl`.
- `::main` es obligatorio porque el planner tiene 2 entrypoints (`main` y `generate`).

### 2. Smoke test — render (FLUX.2, N7)

```powershell
modal run modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
```
- Genera **4 variaciones** del prompt (`outputs/<id>_0..3.webp`); cambialo con `--variations N`.
- Prompt negativo opcional: `--negative default` (FLUX necesita CFG real → ~2× coste de render).
- 1ª vez **descarga FLUX.2-klein (~29 GB)**.

### 3. Pipeline completo N1→N7 (requiere deploy antes)

```powershell
modal deploy modal_app/planner.py
modal deploy modal_app/renderer.py
modal run    modal_app/pipeline.py --prompt "a roadside diner on a freshly terraformed moon"
```
- El `deploy` publica las apps; el pipeline las busca por nombre (`Cls.from_name`).
- Devuelve la traza completa + imagen + **coste e2e** (razonamiento + render).

### 4. Generar el dataset (fan-out en paralelo)

**Solo razonamiento (N1–N6, sin imágenes):**
```powershell
modal run modal_app/planner.py::generate --limit 5      # empezá chico para ver el coste
modal run modal_app/planner.py::generate --limit 100    # Smoke 100 (expansión estratificada)
modal run modal_app/planner.py::generate                # los 36 prompts semilla curados
```

**Dataset completo N1→N7 (CON imágenes ligadas a la traza)** — requiere el deploy del paso 3:
```powershell
modal run modal_app/dataset.py::generate_full --limit 5      # prueba: 5 muestras × 4 imgs
modal run modal_app/dataset.py::generate_full --limit 100    # Smoke 100 (~400 imágenes, ≈ $2.1)
```
- Por muestra: traza N1–N6 **+ 4 imágenes** (`{trace.id}_0..3.webp`) con `sha256` por variación + **seed determinista** (`derive_seed(prompt)` → `meta.seed`/`images[].seed`, **reproducible bit-a-bit**) + bloque `dataset` (licencia + git sha + estrato).
- El dataset completo queda en `outputs/dataset.jsonl` (local); las **imágenes** en el Volume `vcot-outputs` (ya **no** se devuelven por la red → escala a 10k+).
- ⚠️ **Nombre `dataset.jsonl` a propósito**: el planner escribe `traces.jsonl` (solo razonamiento) en el Volume, así que `modal volume get` **no debe** pisar tu dataset completo. Por eso van en archivos distintos.
- Para traer las imágenes a local (para pack): `modal volume get vcot-outputs / outputs` (no pisa `dataset.jsonl`).
- **Recuperación**: si perdés/pisás `dataset.jsonl`, se reconstruye desde el Volume (`records.jsonl` + `{id}.trace.json`):
  ```powershell
  python -m vcot.dataset.assemble --records outputs/records.jsonl --traces-dir outputs --out outputs/dataset.jsonl
  ```

### 4b. Evaluar el dataset (research-grade: CLIP + aesthetic + faithfulness + NSFW + dedup)

```powershell
modal run modal_app/eval.py::evaluate                   # sobre outputs/dataset.jsonl
```
- Puntúa cada muestra: **CLIP** (alineación prompt↔imagen), **ImageReward** (preferencia humana), **aesthetic**, **layout-faithfulness** por IoU vs el scene-graph (con `detection_coverage` que separa "no detectado" de "mal ubicado") y **NSFW**; deduplica por **pHash DCT**.
- Lee las imágenes del **Volume montado** (no hace falta `volume get` para evaluar).
- Escribe `outputs/dataset.eval.jsonl` (bloque `dataset`: quality/safety/split) + `outputs/quality.json` (distribuciones p10/p50/p90 + breakdown por estrato + provenance de modelos).
- Corre en `L4`. Cada scorer está guardado: si un modelo no carga, su score queda `null` sin tumbar la corrida.

> **Faithfulness = línea base.** El render usa solo prompt-enrichment (el layout no condiciona el píxel), así que la métrica mide el *gap*, no un mecanismo de control. Por eso se reporta junto a `detection_coverage`.

### 4b-cal. (Opcional) Calibrar los umbrales del gate contra juicio humano

```powershell
python -m vcot.eval.calibration sheet outputs/dataset.eval.jsonl --out outputs/label_sheet.csv --n 100
# … completá la columna human_good con 1 (buena) / 0 (mala) …
python -m vcot.eval.calibration calibrate outputs/label_sheet.csv
```
- Reporta la correlación de Spearman de cada métrica con el juicio humano y un umbral sugerido (máx F1). Sin esto, los umbrales del gate son arbitrarios.

### 4c. Empaquetar el dataset (WebDataset shards + datacard)

```powershell
modal volume get vcot-outputs / outputs              # traer las imágenes a local (si no lo hiciste)
python -m vcot.dataset.pack outputs/dataset.eval.jsonl --images outputs --out dataset/
```
- Genera `dataset/shards/*.tar` (por muestra: `{id}.json` + `{id}.{idx}.webp`), `index.jsonl` (con split/estrato/seed/métricas/safety), `manifest.json` (versión, git sha, modelos, splits, estratos, seed policy real) y `DATACARD.md` (Datasheet for Datasets, con sección **Safety**).
- `--only-passed` empaqueta solo las muestras que pasan el gate.

### 5. Informe final profesional (Markdown + JSON + CSV con fecha)

```powershell
modal volume get vcot-outputs container_costs.jsonl outputs/   # coste REAL (opcional pero recomendado)
python -m vcot.reporting outputs/ --out reports/
```
- Escribe `reports/<timestamp>/` con `report.md`, `report.json`, `cost_by_stage.csv`, `runs.csv` y actualiza `reports/latest.md`.
- Si trajiste `container_costs.jsonl`, la **§6 "Coste real facturado"** suma el gasto real medido (vida completa + CPU/mem) y el ratio real/marginal; sin él, el informe usa el coste real **estimado** del ledger.
- Vista rápida sin bundle: `python -m vcot.analysis.aggregate outputs/traces.jsonl`

### 6. Preparar la destilación de "Klein" (local) y entrenar (GPU)

```powershell
python -m vcot.train.distill outputs/traces.jsonl --out outputs/sft.jsonl
python -m vcot.train.distill outputs/traces.jsonl --out outputs/sft.jsonl --train   # requiere: pip install -e ".[train]"
```

---

## Resumen ultra-corto (copia-pega)

```powershell
.\.venv\Scripts\Activate.ps1
modal run    modal_app/planner.py::main --prompt "a lone astronaut in a gothic cathedral, moonlight"
modal run    modal_app/renderer.py --prompt "a lone astronaut in a gothic cathedral, moonlight"
modal deploy modal_app/planner.py
modal deploy modal_app/renderer.py
modal run    modal_app/pipeline.py --prompt "a roadside diner on a freshly terraformed moon"
modal run    modal_app/dataset.py::generate_full --limit 5      # dataset con imágenes (prueba)
modal run    modal_app/dataset.py::generate_full --limit 100    # Smoke 100 (~400 imágenes)
python -m vcot.reporting outputs/ --out reports/
python -m vcot.train.distill outputs/traces.jsonl --out outputs/sft.jsonl
```

---

## Variantes útiles

```powershell
# Modelo de razonamiento más grande (necesita A100-80GB)
$env:VCOT_PLANNER_MODEL = "qwen3-14b"; $env:VCOT_PLANNER_GPU = "A100-80GB"

# Render de máxima calidad (FLUX.2-dev 32B, B200 + offload)
$env:VCOT_MODEL = "dev"

# Dataset desde tu propio archivo de prompts (uno por línea)
modal run modal_app/planner.py::generate --prompts-file mis_prompts.txt

# Probar el razonamiento GRATIS en local (sin Modal) con Ollama
ollama serve; ollama pull qwen3:8b
python -m vcot.pipeline.run --prompt "..."
```

---

## Qué queda registrado (auditoría)

| Archivo | Qué es |
|---|---|
| `outputs/traces.jsonl` | dataset de trazas (pensamiento visual) — acumulado en cada run |
| `outputs/runs.jsonl` | ledger: cada proceso (modelo, GPU, ítems, coste **marginal + real estimado**, duración, estado) |
| `outputs/container_costs.jsonl` (Volume) | **coste REAL facturado** por contenedor (vida completa + CPU/mem del cgroup); la verdad del gasto |
| `outputs/<id>.trace.json` / `<id>.webp` | traza/imagen individuales |
| `reports/<timestamp>/` | informe final con fecha (md + json + csv) |
| `reports/latest.md` | último informe, siempre actualizado |

---

## Coste: marginal vs REAL facturado

⚠️ **Importante:** los `$/muestra` de abajo son el coste **marginal** — solo la
inferencia, lo que mide `cost_timer`. **No es lo que pagás.** Modal factura la **vida
completa del contenedor**: carga del modelo (`@modal.enter`) + inferencias + el **idle
de `scaledown_window` (120 s)** antes de apagarse, más CPU y memoria. Para una corrida
**dispersa** (una sola llamada) el coste real es ~**16×** el marginal ($0.0083 → ~$0.13
en un render klein); se amortiza hacia 1× solo si un contenedor cálido sirve **muchas**
llamadas (por eso el dataset usa fan-out `.map`).

### Coste marginal medido (por imagen / traza) — Modal, 2026-06-25

| Paso | GPU | compute (media) | $/muestra (marginal) | carga (1ª vez) |
|---|---|--:|--:|--:|
| Razonamiento N1–N6 (Qwen3-8B) | A100-40GB | 24.1 s | $0.0141 (rango 0.0074–0.0268) | 39–76 s |
| Render N7 (FLUX.2-klein, 4 variaciones) | A100-80GB | 9.6 s | $0.0067 ($0.0017/img) | ~149 s (descarga ~29 GB) |
| **E2E marginal (razonamiento + 4 imgs)** | — | — | **≈ $0.021** | — |

El razonamiento cuesta ~2× el render (4 imgs) y la etapa N2 (layout) es ~50% del coste. Detalle (dataset de 15 trazas) en [IDEA.md §8.3](IDEA.md).

### Ver el coste REAL facturado

Cada contenedor escribe su coste real a `container_costs.jsonl` en el Volume — lo mide
`ContainerMeter` en `@modal.exit` (vida completa × tarifa GPU+CPU+memoria, con CPU y
memoria **leídos del cgroup**, no estimados). Es la **verdad** de tu gasto.

```powershell
modal volume get vcot-outputs container_costs.jsonl outputs/   # traer la verdad a local
python -m vcot.reporting outputs/ --out reports/               # §6 "Coste real facturado" lo suma
```

- El informe muestra el **total real**, el **ratio real/marginal** y el desglose GPU/CPU/mem.
- Los entrypoints (`planner::main`, `renderer`) ya imprimen al terminar **marginal vs real estimado** (`projected_container_cost`: carga + activo + idle); `runs.jsonl` guarda ambos (`total_cost_usd` marginal + `real_cost_est_usd`).
- **Bajar el coste real:** batch denso (`.map` / `generate_full`), bajar `scaledown_window` para corridas sueltas, y preferir A100-80GB sobre B200 para klein.

## Gotchas

1. **Consola Windows + CLI de modal:** el CLI imprime `✓` (UTF-8) y la consola es cp1252 → crashea. Antes de cualquier `modal run`, ejecutá: `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"`.
2. **Costes:** todo se factura por segundo de **vida del contenedor** (no solo la inferencia): incluye la **descarga/carga del modelo**, el **idle de 120 s** tras la última llamada, y CPU+memoria. El `$/muestra` reportado es **marginal**; el real es bastante mayor en corridas sueltas (~16×) → mirá `container_costs.jsonl` / la §6 del informe. Empezá con `--limit` chico y poné un *spend limit* en el dashboard de Modal.
3. **`pipeline.py` necesita `modal deploy`** de planner y renderer antes (paso 3).
4. **Primeras corridas lentas** por la descarga de modelos (~16 GB Qwen3 + ~29 GB FLUX); solo pasa una vez (quedan en los Volumes).
5. **HF token:** Qwen3 es público (corre sin token), pero **FLUX.2 es gated** → el secret `huggingface-secret` con `HF_TOKEN` válido es obligatorio para el render.

---

## Setup (si te falta algo)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[modal]"
modal setup                                              # autenticar (abre el navegador)
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxx
```
Aceptá la licencia gated de FLUX.2-klein en HuggingFace antes del paso 2.
