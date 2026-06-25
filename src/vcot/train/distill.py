"""Destilación de Klein: prep de datos + SFT (IDEA.md §5.1, M5).

`build_sft_dataset` (stdlib, testeable) transforma ``traces.jsonl`` en ejemplos de
entrenamiento. `train` ejecuta el fine-tuning supervisado (lazy-import de TRL),
pensado para correr en una GPU de Modal. Por defecto el CLI solo **prepara** los
datos; con ``--train`` lanza el entrenamiento.

La tesis (IDEA.md §1.2): un modelo pequeño que aprende a *generar la cadena de
decisiones* debería superar, a igualdad de parámetros, a uno que solo mapea
``prompt → imagen``. Este módulo deja el experimento E6 listo para ejecutarse.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from vcot.analysis.aggregate import load_jsonl
from vcot.dataset.sft import trace_to_sft, trace_to_token_target
from vcot.pipeline.schemas import VCoTTrace


def build_sft_dataset(traces_path: str, out_path: str, *, fmt: str = "messages") -> int:
    """Convierte ``traces.jsonl`` → dataset SFT. Devuelve el nº de ejemplos.

    ``fmt`` ∈ {"messages" (chat: prompt → razonamiento completo),
    "tokens" (prompt → secuencia de Visual Tokens)}.
    """
    if fmt not in ("messages", "tokens"):
        raise ValueError(f"fmt no válido: {fmt!r} (usa 'messages' o 'tokens')")

    builder = trace_to_sft if fmt == "messages" else trace_to_token_target
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for record in load_jsonl(traces_path):
            trace = VCoTTrace.model_validate(record)
            out.write(json.dumps(builder(trace), ensure_ascii=False) + "\n")
            n += 1
    return n


def train(
    sft_path: str,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "outputs/klein-sft",
    *,
    epochs: int = 1,
) -> None:  # pragma: no cover - requiere GPU + extra `train`
    """Fine-tuning supervisado de Klein sobre el dataset V-CoT (extra ``train``).

    Lazy-import de TRL/transformers para no cargar el stack pesado salvo cuando se
    entrena de verdad. Pensado para lanzarse en una GPU de Modal.
    """
    from datasets import load_dataset  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    ds = load_dataset("json", data_files=sft_path, split="train")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds,
        args=SFTConfig(output_dir=output_dir, num_train_epochs=epochs, bf16=True),
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Klein destilado -> {output_dir}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vcot-distill",
        description="Prepara el dataset SFT de Klein (y opcionalmente entrena).",
    )
    parser.add_argument("traces", help="Ruta a traces.jsonl")
    parser.add_argument("--out", default="outputs/sft.jsonl", help="JSONL de salida.")
    parser.add_argument("--format", choices=["messages", "tokens"], default="messages")
    parser.add_argument("--train", action="store_true", help="Lanzar el entrenamiento SFT.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", default="outputs/klein-sft")
    args = parser.parse_args(argv)

    n = build_sft_dataset(args.traces, args.out, fmt=args.format)
    print(f"{n} ejemplos SFT ({args.format}) -> {args.out}")

    if args.train:
        train(args.out, base_model=args.base_model, output_dir=args.output_dir)
    else:
        print("(prep-only; usa --train para entrenar, requiere GPU + extra `train`)")


if __name__ == "__main__":
    main()
