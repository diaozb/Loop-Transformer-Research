from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ltf.config import RunConfig, save_resolved_config


class RunLogger:
    def __init__(self, config: RunConfig):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        configured_name = getattr(config, "run_name", None)
        if configured_name:
            safe_configured_name = _safe_name(configured_name)
            seed_token = f"seed{config.seed}"
            suffix = "" if seed_token in safe_configured_name else f"_{seed_token}"
            run_name = f"{stamp}_{safe_configured_name}{suffix}"
        else:
            run_name = f"{stamp}_{config.task.name}_{config.trainer.name}_seed{config.seed}"
        self.run_dir = Path(config.logging.output_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_handle = self.metrics_path.open("a", encoding="utf-8")
        save_resolved_config(config, self.run_dir / "config_resolved.yaml")
        self._write_metadata()

    def log_metrics(self, step: int, metrics: Dict[str, Any]) -> None:
        row = {"step": step, **metrics}
        self._metrics_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._metrics_handle.flush()

    def close(self) -> None:
        self._metrics_handle.close()

    def _write_metadata(self) -> None:
        metadata = {
            "argv": sys.argv,
            "git_status": _run_git(["git", "status", "--short"]),
            "git_head": _run_git(["git", "rev-parse", "HEAD"]),
        }
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def _run_git(cmd):
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), check=False, text=True, capture_output=True)
    except OSError as exc:
        return f"unavailable: {exc}"
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
