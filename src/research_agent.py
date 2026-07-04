from __future__ import annotations

import argparse
import json
import random
from typing import Any

from config_utils import load_config, load_yaml
from logger import load_leaderboard


CHANGE_MAP = {
    "learning_rate": "training.learning_rate",
    "lora_r": "lora.r",
    "lora_alpha": "lora.alpha",
    "lora_dropout": "lora.dropout",
    "max_steps": "training.max_steps",
    "max_seq_length": "training.max_seq_length",
    "gradient_accumulation_steps": "training.gradient_accumulation_steps",
}


def experiment_signature(changes: dict[str, Any]) -> str:
    return json.dumps(changes, sort_keys=True)


def used_signatures(leaderboard: list[dict[str, Any]]) -> set[str]:
    return {experiment_signature(row.get("config_changes", {})) for row in leaderboard}


def propose_config_change(
    base_config: dict[str, Any],
    leaderboard: list[dict[str, Any]],
    search_space: dict[str, list[Any]],
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed + len(leaderboard))
    used = used_signatures(leaderboard)
    last = leaderboard[-1] if leaderboard else {}
    candidates: list[tuple[str, dict[str, Any]]] = []
    if not leaderboard:
        candidates.append(("Start with the default LoRA configuration as the baseline controlled experiment.", {}))
    if last.get("syntax_validity", 1.0) < 0.6:
        candidates.extend(
            [
                ("Lower the learning rate to improve SQL syntax stability.", {"training.learning_rate": min(search_space["learning_rate"])}),
                ("Train for more steps to improve instruction following before changing capacity.", {"training.max_steps": max(search_space["max_steps"])}),
            ]
        )
    if last.get("easy_accuracy", 0.0) > last.get("hard_accuracy", 0.0) + 0.15:
        candidates.extend(
            [
                ("Increase LoRA rank to improve adaptation capacity for harder multi-join SQL.", {"lora.r": max(search_space["lora_r"]), "lora.alpha": max(search_space["lora_alpha"])}),
                ("Increase max sequence length so harder schema prompts are less likely to be truncated.", {"training.max_seq_length": max(search_space["max_seq_length"])}),
            ]
        )
    if last.get("execution_accuracy", 0.0) < 0.3:
        candidates.append(("Try a conservative dropout setting to reduce noisy adaptation.", {"lora.dropout": 0.05}))
    for raw_key, values in search_space.items():
        dotted = CHANGE_MAP[raw_key]
        value = rng.choice(values)
        candidates.append((f"Explore {dotted}={value} as a single controlled configuration change.", {dotted: value}))
    for hypothesis, changes in candidates:
        if experiment_signature(changes) not in used:
            return {"hypothesis": hypothesis, "changes": changes}
    fallback = {
        "training.seed": seed + len(leaderboard) + 1,
    }
    return {
        "hypothesis": "Repeat the best-known configuration with a different seed to check result stability.",
        "changes": fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_coder_3b_lora.yaml")
    parser.add_argument("--search-space", default="configs/search_space.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    leaderboard = load_leaderboard(config["paths"]["leaderboard_file"])
    proposal = propose_config_change(config, leaderboard, load_yaml(args.search_space), seed=int(config["project"]["seed"]))
    print(json.dumps(proposal, indent=2))


if __name__ == "__main__":
    main()
