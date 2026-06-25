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

```powershell
modal run modal_app/planner.py::generate --limit 5      # empezá chico para ver el coste
modal run modal_app/planner.py::generate                # los 36 prompts semilla
```
- Acumula `outputs/traces.jsonl` (local) **y** el Volume `vcot-outputs`.
- Registra la corrida (ítems, coste, duración, estado) en `outputs/runs.jsonl`.

### 5. Informe final profesional (Markdown + JSON + CSV con fecha)

```powershell
python -m vcot.reporting outputs/ --out reports/
```
- Escribe `reports/<timestamp>/` con `report.md`, `report.json`, `cost_by_stage.csv`, `runs.csv` y actualiza `reports/latest.md`.
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
modal run    modal_app/planner.py::generate --limit 5
modal run    modal_app/planner.py::generate
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
| `outputs/runs.jsonl` | ledger: cada proceso (modelo, GPU, ítems, coste, duración, estado) |
| `outputs/<id>.trace.json` / `<id>.webp` | traza/imagen individuales |
| `reports/<timestamp>/` | informe final con fecha (md + json + csv) |
| `reports/latest.md` | último informe, siempre actualizado |

---

## Tiempos y costes reales (medidos en Modal, 2026-06-25)

| Paso | GPU | compute | $/muestra | carga (1ª vez) |
|---|---|--:|--:|--:|
| Razonamiento N1–N6 (Qwen3-8B) | A100-40GB | 12.6–18.9 s | $0.0074–0.011 | 39–76 s |
| Render N7 (FLUX.2-klein) | A100-80GB | 3.19 s | $0.00221 | ~149 s (descarga ~29 GB) |
| **E2E N1→N7** | — | — | **≈ $0.0096** | — |

El razonamiento cuesta ~3× el render; la etapa N2 (layout) es la más cara. Detalle en [IDEA.md §8.3](IDEA.md).

## Gotchas

1. **Consola Windows + CLI de modal:** el CLI imprime `✓` (UTF-8) y la consola es cp1252 → crashea. Antes de cualquier `modal run`, ejecutá: `$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"`.
2. **Costes:** todo se factura por segundo de GPU activa, **incluida la descarga del modelo** en el primer run. Empezá con `--limit` chico y poné un *spend limit* en el dashboard de Modal.
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
