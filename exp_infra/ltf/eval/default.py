from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .results_io import write_rows_csv


def write_default_eval_outputs(output_dir: str | Path, row: Mapping[str, object]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval_metrics.json").write_text(
        json.dumps(dict(row), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_rows_csv(output / "eval_metrics.csv", [row])
