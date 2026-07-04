from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from utils import project_path


def read_schema(db_path: str | Path) -> dict[str, Any]:
    target = project_path(db_path)
    con = sqlite3.connect(target)
    con.row_factory = sqlite3.Row
    tables = [
        row["name"]
        for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    schema: dict[str, Any] = {"database": target.stem, "tables": {}, "foreign_keys": []}
    for table in tables:
        columns = []
        for col in con.execute(f"PRAGMA table_info({table})"):
            parts = [col["name"], col["type"] or "TEXT"]
            if col["pk"]:
                parts.append("PRIMARY KEY")
            if col["notnull"]:
                parts.append("NOT NULL")
            columns.append({"name": col["name"], "type": col["type"], "definition": " ".join(parts)})
        schema["tables"][table] = {"columns": columns}
        for fk in con.execute(f"PRAGMA foreign_key_list({table})"):
            schema["foreign_keys"].append(
                {
                    "table": table,
                    "from": fk["from"],
                    "ref_table": fk["table"],
                    "to": fk["to"],
                }
            )
    con.close()
    return schema


def schema_context(
    db_path: str | Path,
    include_sample_rows: bool = False,
    sample_rows_per_table: int = 2,
) -> str:
    target = project_path(db_path)
    schema = read_schema(target)
    lines = [f"Database: {schema['database']}", ""]
    con = sqlite3.connect(target)
    con.row_factory = sqlite3.Row
    for table, payload in schema["tables"].items():
        lines.append(f"Table: {table}")
        lines.append("Columns:")
        for col in payload["columns"]:
            lines.append(f"- {col['definition']}")
        if include_sample_rows:
            rows = con.execute(f"SELECT * FROM {table} LIMIT ?", (sample_rows_per_table,)).fetchall()
            if rows:
                lines.append("Sample rows:")
                for row in rows:
                    lines.append(f"- {dict(row)}")
        lines.append("")
    if schema["foreign_keys"]:
        lines.append("Foreign keys:")
        for fk in schema["foreign_keys"]:
            lines.append(f"{fk['table']}.{fk['from']} -> {fk['ref_table']}.{fk['to']}")
    con.close()
    return "\n".join(lines).strip()


def save_schema_context(db_path: str | Path, output_path: str | Path) -> None:
    target = project_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(schema_context(db_path), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--include-sample-rows", action="store_true")
    args = parser.parse_args()
    print(schema_context(args.db_path, include_sample_rows=args.include_sample_rows))


if __name__ == "__main__":
    main()
