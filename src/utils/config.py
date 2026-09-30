"""Configuration loading with dot-path overrides."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


SELECTED_CONFIG = PROJECT_ROOT / "configs" / "selected.yaml"
WORKSPACE_PATHS = ("processed_dir", "validation_dir", "experiments_dir", "models_dir", "runs_dir", "output_dir")


def selected_path() -> Path:
    """configs/selected.yaml, or <workspace>/selected.yaml when ENTIQ_WORKSPACE isolates a sandbox run."""
    ws = os.environ.get("ENTIQ_WORKSPACE")
    return Path(ws) / "selected.yaml" if ws else SELECTED_CONFIG


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None,
                use_selected: bool = True) -> dict:
    """Base config, then configs/selected.yaml (choices made by validation experiments), then overrides."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    sel = selected_path()
    if use_selected and path is None and sel.exists():
        with open(sel, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, (yaml.safe_load(fh) or {}).get("overrides", {}))
    cfg = copy.deepcopy(cfg)
    ws = os.environ.get("ENTIQ_WORKSPACE")
    if ws:  # sandbox: every generated artefact lives under the workspace
        for k in WORKSPACE_PATHS:
            cfg["paths"][k] = str(Path(ws) / Path(cfg["paths"][k]).name)
    for k, v in (overrides or {}).items():
        _set_dotted(cfg, k, v)
    return cfg


def save_selection(section: str, overrides: dict, evidence: dict) -> None:
    """Record an experiment-driven choice (with its evidence) in configs/selected.yaml."""
    data = {"overrides": {}, "evidence": {}}
    target = selected_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or data
    data.setdefault("overrides", {})
    data.setdefault("evidence", {})
    data["overrides"] = deep_merge(data["overrides"], overrides)
    data["evidence"][section] = evidence
    header = "# Written by experiments (src/cli.py). Choices here were selected on validation data.\n"
    target.write_text(header + yaml.safe_dump(json.loads(json.dumps(data, default=_plain)), sort_keys=False),
                      encoding="utf-8")


def _plain(o):
    try:
        return o.item()  # numpy scalars
    except Exception:
        return str(o)


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a `paths.*` entry relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def config_hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def parse_override(expr: str) -> tuple[str, Any]:
    """Parse `a.b.c=value` where value is YAML (so numbers/bools/lists work)."""
    key, _, raw = expr.partition("=")
    return key.strip(), yaml.safe_load(raw)
