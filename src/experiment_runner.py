from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from config_utils import apply_changes, load_config, load_yaml, save_yaml
from evaluate_sql import evaluate_examples, save_report
from infer_sql import generate_predictions
from logger import (
    append_leaderboard,
    append_research_log,
    is_new_best,
    load_leaderboard,
    make_leaderboard_row,
)
from research_agent import propose_config_change
from train_lora import train_lora
from utils import ensure_dir, next_experiment_id, project_path, read_jsonl


def copy_best(adapter_path: Path, config_path: Path, best_dir: Path) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    target_adapter = best_dir / "adapter"
    if target_adapter.exists():
        shutil.rmtree(target_adapter)
    if adapter_path.exists():
        shutil.copytree(adapter_path, target_adapter)
    shutil.copy2(config_path, best_dir / "config.yaml")


def synthetic_dry_report() -> dict[str, Any]:
    return {
        "mode": "dry_run",
        "total": 0,
        "execution_accuracy": 0.0,
        "syntax_validity": 0.0,
        "exact_result_match": 0.0,
        "timeout_count": 0,
        "error_count": 0,
        "easy_accuracy": 0.0,
        "medium_accuracy": 0.0,
        "hard_accuracy": 0.0,
        "ecommerce_accuracy": 0.0,
        "saas_accuracy": 0.0,
        "banking_accuracy": 0.0,
    }


def run_one_experiment(base_config: dict[str, Any], search_space: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    leaderboard = load_leaderboard(base_config["paths"]["leaderboard_file"])
    experiment_id = next_experiment_id(base_config["paths"]["experiments_dir"])
    proposal = propose_config_change(base_config, leaderboard, search_space, seed=int(base_config["project"]["seed"]))
    experiment_config = apply_changes(base_config, proposal["changes"])
    exp_dir = ensure_dir(Path(base_config["paths"]["experiments_dir"]) / experiment_id)
    config_path = exp_dir / "config.yaml"
    save_yaml(config_path, experiment_config)
    (exp_dir / "hypothesis.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    if dry_run:
        adapter_path = exp_dir / "adapter"
        adapter_path.mkdir(exist_ok=True)
        report = synthetic_dry_report()
        save_report(base_config, report, [], experiment_id)
        training_summary = {"dry_run": True}
    else:
        adapter_path = train_lora(experiment_config, experiment_id)
        examples = read_jsonl(experiment_config["paths"]["eval_file"])
        predictions = generate_predictions(experiment_config, examples, adapter_path=str(adapter_path))
        report, details = evaluate_examples(experiment_config, examples, predictions, mode="adapter")
        save_report(experiment_config, report, details, experiment_id)
        metrics_path = exp_dir / "training_metrics.json"
        training_summary = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    fresh_leaderboard = load_leaderboard(base_config["paths"]["leaderboard_file"])
    candidate = make_leaderboard_row(
        experiment_config,
        experiment_id,
        str(adapter_path),
        proposal["hypothesis"],
        proposal["changes"],
        report,
        "pending",
    )
    decision = "kept" if is_new_best(fresh_leaderboard, candidate) else "rejected"
    candidate["decision"] = decision
    append_leaderboard(base_config, candidate)
    append_research_log(
        base_config,
        experiment_id,
        proposal["hypothesis"],
        proposal["changes"],
        training_summary,
        report,
        decision,
    )
    if decision == "kept" and not dry_run:
        copy_best(Path(adapter_path), config_path, project_path(base_config["paths"]["best_dir"]))
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen25_coder_3b_lora.yaml")
    parser.add_argument("--search_space", default="configs/search_space.yaml")
    parser.add_argument("--n_experiments", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    base_config = load_config(args.config)
    search_space = load_yaml(args.search_space)
    results = []
    for _ in range(args.n_experiments):
        result = run_one_experiment(base_config, search_space, dry_run=args.dry_run)
        results.append(result)
        print(json.dumps(result, indent=2))
    print(f"Completed {len(results)} experiment(s).")


if __name__ == "__main__":
    main()
