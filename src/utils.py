from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    target = project_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = project_path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")


def flatten_result(rows: Iterable[Iterable[Any]]) -> list[list[Any]]:
    normalized: list[list[Any]] = []
    for row in rows:
        normalized.append([normalize_value(value) for value in row])
    return normalized


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)


def next_experiment_id(experiments_dir: str | Path = "experiments") -> str:
    root = ensure_dir(experiments_dir)
    existing = []
    for path in root.glob("exp_*"):
        if path.is_dir():
            suffix = path.name.replace("exp_", "", 1)
            if suffix.isdigit():
                existing.append(int(suffix))
    return f"exp_{(max(existing) + 1) if existing else 1:03d}"
