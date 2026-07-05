from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from typing import Any

from utils import read_jsonl


TABLE_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.I)
VARIANT_WRAPPER = re.compile(r"^\s*SELECT\s+\*\s+FROM\s+\((.*)\)\s+AS\s+report_variant_\d+\s*$", re.I | re.S)


def unwrap_variant(sql: str) -> str:
    match = VARIANT_WRAPPER.match(sql.strip())
    return match.group(1).strip() if match else sql


def sql_features(sql: str) -> dict[str, Any]:
    sql = unwrap_variant(sql)
    upper = sql.upper()
    tables = sorted(set(TABLE_PATTERN.findall(sql)))
    return {
        "table_count": len(tables),
        "tables": tables,
        "join_count": upper.count(" JOIN "),
        "has_cte": upper.strip().startswith("WITH ") or " WITH " in upper,
        "has_window": " OVER " in upper or " OVER(" in upper,
        "has_subquery": upper.count("SELECT") > 1,
        "has_group_by": " GROUP BY " in upper,
        "has_having": " HAVING " in upper,
        "has_order_limit": " ORDER BY " in upper and " LIMIT " in upper,
    }


def audit(rows: list[dict[str, Any]], label: str) -> None:
    print(f"\nDataset audit: {label}")
    print(f"examples: {len(rows)}")
    by_db = Counter(row["db_id"] for row in rows)
    by_difficulty = Counter(row["difficulty"] for row in rows)
    print(f"by db: {dict(by_db)}")
    print(f"by difficulty: {dict(by_difficulty)}")
    feature_rows = [sql_features(row["gold_sql"]) for row in rows]
    if not feature_rows:
        return
    avg_tables = sum(row["table_count"] for row in feature_rows) / len(feature_rows)
    avg_joins = sum(row["join_count"] for row in feature_rows) / len(feature_rows)
    print(f"avg tables/query: {avg_tables:.2f}")
    print(f"avg joins/query: {avg_joins:.2f}")
    for feature in ["has_cte", "has_window", "has_subquery", "has_group_by", "has_having", "has_order_limit"]:
        count = sum(1 for row in feature_rows if row[feature])
        print(f"{feature}: {count} ({count / len(feature_rows) * 100:.1f}%)")
    hard_rows = [row for row in rows if row["difficulty"] == "hard"]
    hard_features = [sql_features(row["gold_sql"]) for row in hard_rows]
    if hard_features:
        avg_hard_tables = sum(row["table_count"] for row in hard_features) / len(hard_features)
        avg_hard_joins = sum(row["join_count"] for row in hard_features) / len(hard_features)
        complex_hard = sum(
            1
            for row in hard_features
            if row["table_count"] >= 3 or row["has_cte"] or row["has_window"] or row["has_subquery"]
        )
        print("\nHard subset")
        print(f"hard examples: {len(hard_rows)}")
        print(f"avg hard tables/query: {avg_hard_tables:.2f}")
        print(f"avg hard joins/query: {avg_hard_joins:.2f}")
        print(f"complex hard examples: {complex_hard} ({complex_hard / len(hard_rows) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--eval", default="data/eval.jsonl")
    args = parser.parse_args()
    audit(read_jsonl(args.train), "train")
    audit(read_jsonl(args.eval), "eval")


if __name__ == "__main__":
    main()
