from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "input_dir": "data/input",
        "workdir": "data/work",
        "output_dir": "data/output",
        "prompt_dir": "prompts",
    },
    "chunking": {
        "max_chars": 6000,
        "overlap_paragraphs": 0,
    },
    "model": {
        "backend": "api",
        "name": "gpt-4.1-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0,
        "max_tokens": 4000,
        "json_mode": True,
        "local_model_path": "",
    },
}


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        loaded = _load_yaml_like(text)
    return deep_merge(DEFAULT_CONFIG, loaded)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(value: str, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_yaml_like(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML config must be a mapping.")
        return data
    except ModuleNotFoundError:
        return _parse_basic_yaml(text)


def _parse_basic_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if raw_value == "":
            new_map: Dict[str, Any] = {}
            current[key] = new_map
            stack.append((indent + 2, new_map))
        else:
            current[key] = _parse_scalar(raw_value)
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
