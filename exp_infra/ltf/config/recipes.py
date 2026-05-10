from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .loader import ConfigError, deep_merge, load_partial_config_dict
from .schema import RunConfig, run_config_from_dict


def load_recipe_file(path: str | Path, shared_overrides: Iterable[str] = ()) -> List[tuple[str, RunConfig]]:
    recipe_path = Path(path).resolve()
    with recipe_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "recipes" not in payload or not isinstance(payload["recipes"], list):
        raise ConfigError("Recipe file must contain a `recipes` list")

    defaults = payload.get("defaults", {}) or {}
    output = []
    for idx, recipe in enumerate(payload["recipes"]):
        if not isinstance(recipe, dict):
            raise ConfigError(f"Recipe at index {idx} must be a mapping")
        raw_name = recipe.get("name", f"recipe_{idx}")
        include_payload, inline_payload = _resolve_recipe_payload(recipe_path, recipe)
        merged = deep_merge(include_payload, copy.deepcopy(defaults))
        merged = deep_merge(merged, inline_payload)
        cfg = run_config_from_dict(merged)
        name = raw_name.format(**_flatten_for_format(cfg.to_dict()))
        # Apply name after formatting so logger can use it.
        setattr(cfg, "run_name", name)
        output.append((name, cfg))
    if shared_overrides:
        raise ConfigError("Shared dotlist overrides for recipe files are not implemented yet.")
    return output


def _resolve_recipe_payload(recipe_path: Path, recipe: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    include = recipe.get("include", [])
    if isinstance(include, (str, Path)):
        include = [include]
    merged: Dict[str, Any] = {}
    for item in include:
        include_path = (recipe_path.parent / str(item)).resolve()
        merged = deep_merge(merged, load_partial_config_dict(include_path))
    inline = {k: v for k, v in recipe.items() if k not in ("name", "include")}
    return merged, inline


def _flatten_for_format(payload: Dict[str, Any]) -> Dict[str, Any]:
    class DotDict(dict):
        def __getattr__(self, item):
            return self[item]

    def convert(value):
        if isinstance(value, dict):
            return DotDict({k: convert(v) for k, v in value.items()})
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value

    return convert(payload)
