from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping

import yaml

from .schema import RunConfig, run_config_from_dict


class ConfigError(ValueError):
    pass


def load_config(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> RunConfig:
    payload = load_config_dict(path, overrides=overrides)
    return run_config_from_dict(payload)


def load_config_dict(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> Dict[str, Any]:
    payload = load_partial_config_dict(path, overrides=overrides)
    _validate_payload(payload)
    return payload


def load_partial_config_dict(path: str | os.PathLike[str], overrides: Iterable[str] = ()) -> Dict[str, Any]:
    path = Path(path).resolve()
    payload = _load_yaml_with_includes(path, seen=[])
    for override in overrides:
        _apply_override(payload, override)
    return payload


def save_resolved_config(config: RunConfig, path: str | os.PathLike[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, MutableMapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_with_includes(path: Path, seen: List[Path]) -> Dict[str, Any]:
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ConfigError(f"Cyclic config include: {chain}")
    if not path.exists():
        raise ConfigError(f"Config does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    includes = raw.pop("include", [])
    if isinstance(includes, (str, os.PathLike)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"`include` must be a string or list in {path}")

    merged: Dict[str, Any] = {}
    for item in includes:
        include_path = (path.parent / str(item)).resolve()
        merged = deep_merge(merged, _load_yaml_with_includes(include_path, [*seen, path]))
    return deep_merge(merged, raw)


def _apply_override(payload: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must be key=value, got: {override}")
    raw_key, raw_value = override.split("=", 1)
    keys = [part for part in raw_key.split(".") if part]
    if not keys:
        raise ConfigError(f"Override key is empty: {override}")
    value = yaml.safe_load(raw_value)

    cursor: Dict[str, Any] = payload
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _validate_payload(payload: Dict[str, Any]) -> None:
    required = ["task", "model", "trainer"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    model = payload.get("model", {})
    if model.get("use_wpe", False) and model.get("use_rope", False):
        raise ConfigError("model.use_wpe and model.use_rope cannot both be true")
    if model.get("wpe_mode", "none") not in ("none", "all", "once"):
        raise ConfigError("model.wpe_mode must be one of: none, all, once")

    eval_cfg = payload.get("eval", {})
    if eval_cfg.get("after_train_checkpoint", "best") not in ("best", "last"):
        raise ConfigError("eval.after_train_checkpoint must be one of: best, last")

    optimizer = payload.get("optimizer", {})
    if optimizer.get("lr_schedule", "none") not in ("none", "cosine_after_curriculum"):
        raise ConfigError("optimizer.lr_schedule must be one of: none, cosine_after_curriculum")
