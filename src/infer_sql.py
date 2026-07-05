from __future__ import annotations

import argparse
import re
from typing import Any

from config_utils import load_config
from utils import read_jsonl, write_jsonl


PROMPT_TEMPLATE = """You are a Text-to-SQL assistant.

Given the database schema and the business question, write one valid SQLite SELECT query.

Rules:
- Return SQL only.
- Do not explain.
- Use only tables and columns from the schema.
- Use SQLite syntax.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, or DETACH.

Database:
{db_id}

Schema:
{schema_context}

Question:
{question}

SQL:
"""


def build_prompt(example: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        db_id=example["db_id"],
        schema_context=example["schema_context"],
        question=example["question"],
    )


def extract_sql(text: str) -> str:
    cleaned = re.sub(r"```(?:sql)?", "", text, flags=re.I).replace("```", "").strip()
    match = re.search(r"\b(WITH|SELECT)\b", cleaned, flags=re.I)
    if match:
        cleaned = cleaned[match.start() :]
    if ";" in cleaned:
        cleaned = cleaned.split(";", 1)[0] + ";"
    return cleaned.strip()


def generate_predictions(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    adapter_path: str | None = None,
    batch_size: int = 1,
    num_candidates: int | None = None,
) -> list[dict[str, Any]]:
    from load_model import load_base_model, load_model_with_adapter

    if adapter_path:
        model, tokenizer = load_model_with_adapter(config, adapter_path)
    else:
        model, tokenizer = load_base_model(config)
    eval_cfg = config.get("evaluation", {})
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 256))
    candidate_count = int(num_candidates or eval_cfg.get("num_candidates", 1))
    temperature = float(eval_cfg.get("temperature", 0.3))
    top_p = float(eval_cfg.get("top_p", 0.95))
    predictions = []
    for example in examples:
        prompt = build_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": candidate_count > 1,
            "num_return_sequences": candidate_count,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if candidate_count > 1:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        output_ids = model.generate(**inputs, **generation_kwargs)
        candidates = []
        for output in output_ids:
            raw_output = tokenizer.decode(output[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            candidates.append({"raw_output": raw_output, "sql": extract_sql(raw_output)})
        predictions.append(
            {
                "id": example["id"],
                "db_id": example["db_id"],
                "pred_sql": candidates[0]["sql"] if candidates else "",
                "candidates": candidates,
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_05b_smoke.yaml")
    parser.add_argument("--input", default="data/eval.jsonl")
    parser.add_argument("--output", default="experiments/predictions.jsonl")
    parser.add_argument("--adapter_path")
    parser.add_argument("--num_candidates", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    examples = read_jsonl(args.input)
    predictions = generate_predictions(config, examples, adapter_path=args.adapter_path, num_candidates=args.num_candidates)
    write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
