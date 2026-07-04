from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from config_utils import load_config
from infer_sql import build_prompt
from utils import ensure_dir, read_jsonl


def format_training_text(example: dict[str, Any]) -> str:
    return build_prompt(example) + example["gold_sql"]


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def train_lora(config: dict[str, Any], experiment_id: str) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires datasets, peft, trl, transformers, accelerate, and torch. "
            "Install requirements.txt before training."
        ) from exc
    from load_model import load_base_model

    exp_dir = ensure_dir(Path(config["paths"]["experiments_dir"]) / experiment_id)
    adapter_dir = ensure_dir(exp_dir / "adapter")
    train_rows = read_jsonl(config["paths"]["train_file"])
    if not train_rows:
        raise FileNotFoundError("No training rows found. Run generate_sql_dataset.py first.")
    model, tokenizer = load_base_model(config)
    if config.get("model", {}).get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    dataset = Dataset.from_list([{"text": format_training_text(row)} for row in train_rows])
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    max_seq_length = int(training_cfg["max_seq_length"])
    use_bf16 = bool(training_cfg.get("bf16", False))
    use_fp16 = bool(training_cfg.get("fp16", True)) and not use_bf16
    sft_kwargs = {
        "output_dir": str(exp_dir / "checkpoints"),
        "max_steps": int(training_cfg["max_steps"]),
        "learning_rate": float(training_cfg["learning_rate"]),
        "per_device_train_batch_size": int(training_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training_cfg["gradient_accumulation_steps"]),
        "warmup_ratio": float(training_cfg["warmup_ratio"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "logging_steps": int(training_cfg["logging_steps"]),
        "save_steps": int(training_cfg["save_steps"]),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 0.0)),
        "dataset_text_field": "text",
        "report_to": [],
    }
    sft_signature = inspect.signature(SFTConfig)
    if "max_seq_length" in sft_signature.parameters:
        sft_kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in sft_signature.parameters:
        sft_kwargs["max_length"] = max_seq_length
    sft_config = SFTConfig(**supported_kwargs(SFTConfig, sft_kwargs))

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": sft_config,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "dataset_text_field": "text",
        "max_seq_length": max_seq_length,
    }
    trainer = SFTTrainer(**supported_kwargs(SFTTrainer, trainer_kwargs))
    result = trainer.train()
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = result.metrics
    (exp_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_05b_smoke.yaml")
    parser.add_argument("--experiment_id", default="exp_test")
    args = parser.parse_args()
    config = load_config(args.config)
    adapter_dir = train_lora(config, args.experiment_id)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
