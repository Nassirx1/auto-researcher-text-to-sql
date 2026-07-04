from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config_utils import load_config
from infer_sql import build_prompt
from utils import ensure_dir, read_jsonl


def format_training_text(example: dict[str, Any]) -> str:
    return build_prompt(example) + example["gold_sql"]


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
    sft_config = SFTConfig(
        output_dir=str(exp_dir / "checkpoints"),
        max_steps=int(training_cfg["max_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        max_seq_length=int(training_cfg["max_seq_length"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        fp16=bool(training_cfg.get("fp16", True)),
        dataset_text_field="text",
        report_to=[],
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, peft_config=peft_config, args=sft_config)
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
