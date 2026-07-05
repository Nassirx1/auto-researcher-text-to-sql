from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from logger import best_score_key, load_leaderboard
from utils import project_path, read_jsonl


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def load_predictions(experiment_id: str) -> list[dict[str, Any]]:
    return read_jsonl(Path("experiments") / experiment_id / "predictions.jsonl")


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No leaderboard rows found.")
        return
    first_acc = float(rows[0].get("execution_accuracy", 0.0))
    previous_acc = first_acc
    print("Experiment leaderboard")
    print("id       exec   delta_prev  delta_first  hard   syntax  pass@k  errors  decision  changes")
    for row in rows:
        acc = float(row.get("execution_accuracy", 0.0))
        delta_prev = acc - previous_acc
        delta_first = acc - first_acc
        print(
            f"{row.get('experiment_id', ''):<8} "
            f"{pct(acc):>6} "
            f"{delta_prev * 100:+10.1f} "
            f"{delta_first * 100:+11.1f} "
            f"{pct(row.get('hard_accuracy')):>6} "
            f"{pct(row.get('syntax_validity')):>7} "
            f"{pct(row.get('pass_at_k', row.get('execution_accuracy'))):>7} "
            f"{row.get('error_count', 0):>6} "
            f"{row.get('decision', ''):<8} "
            f"{row.get('config_changes', {})}"
        )
        previous_acc = acc


def show_prediction_samples(experiment_id: str, limit: int) -> None:
    predictions = load_predictions(experiment_id)
    failures = [row for row in predictions if not row.get("correct")]
    successes = [row for row in predictions if row.get("correct")]
    print(f"\nQwen SQL samples from {experiment_id}")
    for label, rows in [("Failures", failures), ("Successes", successes)]:
        print(f"\n{label}:")
        for row in rows[:limit]:
            print(f"- {row['id']} ({row['db_id']}, {row['difficulty']})")
            print(f"  Question: {row.get('question', '[question stored in eval.jsonl]')}")
            print(f"  Gold SQL: {row.get('gold_sql', '')}")
            print(f"  Qwen SQL: {row.get('pred_sql', '')}")
            if row.get("error"):
                print(f"  Error: {row['error']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", default="leaderboard.jsonl")
    parser.add_argument("--experiment_id", help="Experiment to sample; defaults to best by score.")
    parser.add_argument("--sample_limit", type=int, default=5)
    args = parser.parse_args()
    rows = load_leaderboard(project_path(args.leaderboard))
    print_table(rows)
    if not rows:
        return
    best = max(rows, key=best_score_key)
    chosen = args.experiment_id or best["experiment_id"]
    first = rows[0]
    print("\nBest experiment")
    print(json.dumps(best, indent=2))
    print(
        "\nImprovement vs first experiment: "
        f"{(float(best.get('execution_accuracy', 0.0)) - float(first.get('execution_accuracy', 0.0))) * 100:+.1f} percentage points"
    )
    show_prediction_samples(chosen, args.sample_limit)


if __name__ == "__main__":
    main()
