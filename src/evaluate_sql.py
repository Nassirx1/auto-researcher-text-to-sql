from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

try:
    import sqlparse
except ImportError:  # pragma: no cover - exercised only in minimal environments
    sqlparse = None

from config_utils import load_config
from infer_sql import generate_predictions
from utils import ensure_dir, flatten_result, project_path, read_jsonl, write_jsonl


FORBIDDEN = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|PRAGMA|ATTACH|DETACH|REPLACE|VACUUM)\b", re.I)


def is_safe_select(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped:
        return False
    if sqlparse is not None:
        parsed = sqlparse.parse(stripped)
        if len(parsed) != 1:
            return False
        first = parsed[0].token_first(skip_cm=True)
        if first is None or first.value.upper() not in {"SELECT", "WITH"}:
            return False
    else:
        without_trailing_semicolon = stripped[:-1] if stripped.endswith(";") else stripped
        if ";" in without_trailing_semicolon:
            return False
        first_word = without_trailing_semicolon.split(None, 1)[0].upper()
        if first_word not in {"SELECT", "WITH"}:
            return False
    return FORBIDDEN.search(stripped) is None


def execute_sql(db_path: str | Path, sql: str, timeout_seconds: int = 10) -> dict[str, Any]:
    started = time.time()
    if not is_safe_select(sql):
        return {"ok": False, "error": "unsafe_or_non_select", "rows": [], "elapsed": 0.0}
    con = sqlite3.connect(project_path(db_path))
    try:
        con.execute(f"PRAGMA busy_timeout = {timeout_seconds * 1000}")
        rows = con.execute(sql).fetchall()
        elapsed = time.time() - started
        if elapsed > timeout_seconds:
            return {"ok": False, "error": "timeout", "rows": [], "elapsed": elapsed}
        return {"ok": True, "error": None, "rows": flatten_result(rows), "elapsed": elapsed}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "rows": [], "elapsed": time.time() - started}
    finally:
        con.close()


def query_has_order_limit(sql: str) -> bool:
    upper = sql.upper()
    return "ORDER BY" in upper and "LIMIT" in upper


def normalize_result(rows: list[list[Any]], ordered: bool = False) -> list[list[Any]]:
    if ordered:
        return rows
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))


def compare_results(pred_rows: list[list[Any]], gold_rows: list[list[Any]], sql: str) -> bool:
    ordered = query_has_order_limit(sql)
    return normalize_result(pred_rows, ordered) == normalize_result(gold_rows, ordered)


def db_path_for(config: dict[str, Any], db_id: str) -> Path:
    return project_path(config["paths"]["database_dir"]) / f"{db_id}.db"


def evaluate_examples(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]] | None = None,
    mode: str = "gold",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timeout = int(config.get("evaluation", {}).get("timeout_seconds", 10))
    by_id = {row["id"]: row for row in predictions or []}
    details = []
    counts = {
        "total": 0,
        "execution_correct": 0,
        "syntax_valid": 0,
        "exact_match": 0,
        "timeout_count": 0,
        "error_count": 0,
    }
    buckets: dict[str, list[bool]] = {key: [] for key in ["easy", "medium", "hard", "ecommerce", "saas", "banking"]}
    for example in examples:
        counts["total"] += 1
        sql = example["gold_sql"] if mode == "gold" else by_id.get(example["id"], {}).get("pred_sql", "")
        db_path = db_path_for(config, example["db_id"])
        result = execute_sql(db_path, sql, timeout)
        expected = example["expected_result"]
        correct = result["ok"] and compare_results(result["rows"], expected, example["gold_sql"])
        syntax_valid = result["ok"]
        exact_match = sql.strip().rstrip(";") == example["gold_sql"].strip().rstrip(";")
        counts["execution_correct"] += int(correct)
        counts["syntax_valid"] += int(syntax_valid)
        counts["exact_match"] += int(exact_match)
        counts["timeout_count"] += int(result["error"] == "timeout")
        counts["error_count"] += int(not result["ok"])
        buckets[example["difficulty"]].append(correct)
        buckets[example["db_id"]].append(correct)
        details.append(
            {
                "id": example["id"],
                "db_id": example["db_id"],
                "difficulty": example["difficulty"],
                "gold_sql": example["gold_sql"],
                "pred_sql": sql,
                "ok": result["ok"],
                "correct": correct,
                "error": result["error"],
                "result": result["rows"],
                "expected_result": expected,
            }
        )
    total = max(counts["total"], 1)
    report = {
        "mode": mode,
        "total": counts["total"],
        "execution_accuracy": counts["execution_correct"] / total,
        "syntax_validity": counts["syntax_valid"] / total,
        "exact_result_match": counts["execution_correct"] / total,
        "timeout_count": counts["timeout_count"],
        "error_count": counts["error_count"],
    }
    for key, values in buckets.items():
        report[f"{key}_accuracy"] = (sum(values) / len(values)) if values else 0.0
    return report, details


def validate_gold_dataset(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_path = project_path(config["paths"]["train_file"])
    eval_path = project_path(config["paths"]["eval_file"])
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError("Expected train/eval JSONL files. Run generate_sql_dataset.py first.")
    for db_id in ["ecommerce", "saas", "banking"]:
        if not db_path_for(config, db_id).exists():
            raise FileNotFoundError(f"Missing database: {db_id}")
    train = read_jsonl(train_path)
    examples = read_jsonl(eval_path)
    train_ids = {row["id"] for row in train}
    eval_ids = {row["id"] for row in examples}
    duplicate_ids = sorted(train_ids & eval_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate train/eval IDs: {duplicate_ids[:5]}")
    report, details = evaluate_examples(config, examples, mode="gold")
    mismatches = [row["id"] for row in details if not row["correct"]]
    if mismatches:
        raise ValueError(f"Gold SQL mismatches expected results: {mismatches[:10]}")
    report["dataset_valid"] = True
    return report, details


def save_report(config: dict[str, Any], report: dict[str, Any], details: list[dict[str, Any]], experiment_id: str | None) -> Path:
    exp_id = experiment_id or "exp_eval"
    exp_dir = ensure_dir(Path(config["paths"]["experiments_dir"]) / exp_id)
    (exp_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(exp_dir / "predictions.jsonl", details)
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", choices=["gold", "baseline", "adapter"], required=True)
    parser.add_argument("--adapter_path")
    parser.add_argument("--experiment_id")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "gold":
        report, details = validate_gold_dataset(config)
    else:
        examples = read_jsonl(config["paths"]["eval_file"])
        predictions = generate_predictions(config, examples, adapter_path=args.adapter_path if args.mode == "adapter" else None)
        report, details = evaluate_examples(config, examples, predictions, mode=args.mode)
    exp_dir = save_report(config, report, details, args.experiment_id)
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation artifacts to {exp_dir}")


if __name__ == "__main__":
    main()
