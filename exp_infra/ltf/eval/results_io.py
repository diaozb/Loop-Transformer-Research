from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dense_results_to_rows(results: Dict[int, Dict[int, Dict[str, float]]]):
    rows = []
    for length, loop_map in sorted(results.items()):
        for loop, metrics in sorted(loop_map.items()):
            rows.append({"length": length, "loop": loop, **metrics})
    return rows
