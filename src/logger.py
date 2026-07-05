from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import append_jsonl, project_path, read_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_leaderboard(path: str | Path = "leaderboard.jsonl") -> list[dict[str, Any]]:
    return read_jsonl(path)


def append_leaderboard(config: dict[str, Any], row: dict[str, Any]) -> None:
    append_jsonl(config["paths"]["leaderboard_file"], row)


def best_score_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("execution_accuracy", 0.0)),
        float(row.get("hard_accuracy", 0.0)),
        float(row.get("syntax_validity", 0.0)),
        -float(row.get("error_count", 0.0)),
    )


def is_new_best(leaderboard: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
    kept = [row for row in leaderboard if row.get("decision") == "kept"]
    if not kept:
        return True
    return best_score_key(candidate) > max(best_score_key(row) for row in kept)


def make_leaderboard_row(
    config: dict[str, Any],
    experiment_id: str,
    adapter_path: str,
    hypothesis: str,
    config_changes: dict[str, Any],
    report: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "timestamp": utc_now(),
        "model_id": config.get("model", {}).get("model_id"),
        "adapter_path": adapter_path,
        "hypothesis": hypothesis,
        "config_changes": config_changes,
        "execution_accuracy": report.get("execution_accuracy", 0.0),
        "pass_at_k": report.get("pass_at_k", report.get("execution_accuracy", 0.0)),
        "syntax_validity": report.get("syntax_validity", 0.0),
        "easy_accuracy": report.get("easy_accuracy", 0.0),
        "medium_accuracy": report.get("medium_accuracy", 0.0),
        "hard_accuracy": report.get("hard_accuracy", 0.0),
        "error_count": report.get("error_count", 0),
        "decision": decision,
    }


def append_research_log(
    config: dict[str, Any],
    experiment_id: str,
    hypothesis: str,
    config_changes: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    decision: str,
) -> None:
    path = project_path(config["paths"]["research_log_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"## Experiment {experiment_id}",
        "",
        "Hypothesis:",
        hypothesis,
        "",
        "Config changes:",
        f"`{config_changes}`",
        "",
        "Training summary:",
        f"`{training_summary}`",
        "",
        "Evaluation summary:",
        f"`{evaluation_summary}`",
        "",
        "Decision:",
        "Kept as new best." if decision == "kept" else "Rejected.",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
