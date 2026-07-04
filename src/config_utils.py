from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal smoke-test environments
    yaml = None

from utils import PROJECT_ROOT, project_path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = project_path(path)
    with target.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _load_simple_yaml(text)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        if yaml is not None:
            yaml.safe_dump(data, handle, sort_keys=False)
        else:
            json.dump(data, handle, indent=2)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _simple_lines(text: str) -> list[tuple[int, str]]:
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    return lines


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        values = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            values.append(_parse_scalar(lines[index][1][2:]))
            index += 1
        return values, index
    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
        item = lines[index][1]
        key, _, value = item.partition(":")
        index += 1
        if value.strip():
            mapping[key] = _parse_scalar(value)
        else:
            child, index = _parse_block(lines, index, indent + 2)
            mapping[key] = child
    return mapping, index


def _load_simple_yaml(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        return json.loads(stripped)
    parsed, _ = _parse_block(_simple_lines(text), 0, 0)
    return parsed


def load_config(path: str | Path) -> dict[str, Any]:
    target = project_path(path)
    config = load_yaml(target)
    inherits = config.pop("inherits", None)
    if inherits:
        parent = target.parent / inherits
        merged = deep_merge(load_config(parent), config)
    else:
        merged = config
    merged["_config_path"] = str(target.relative_to(PROJECT_ROOT))
    return merged


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def apply_changes(config: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for key, value in changes.items():
        set_nested(updated, key, value)
    return updated
