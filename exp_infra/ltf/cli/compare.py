from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from ltf.config import load_config
from ltf.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal comparison across multiple experiment configs.")
    parser.add_argument("--comparison", required=True, help="Path to comparison YAML.")
    parser.add_argument(
        "--output-dir",
        default="exp_infra/results/comparisons",
        help="Directory where comparison CSV/Markdown summaries are written.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Shared dotlist override applied to every experiment. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison).resolve()
    comparison = _load_comparison(comparison_path)
    rows = []

    for item in comparison["experiments"]:
        name = item["name"]
        config_path = (comparison_path.parent / item["config"]).resolve()
        overrides = list(args.set) + list(item.get("overrides", []))
        config = load_config(config_path, overrides=overrides)
        run_dir = run_training(config)
        row = _summarize_run(name=name, config_path=config_path, run_dir=Path(run_dir), config=config)
        rows.append(row)

    output_root = Path(args.output_dir) / comparison.get("name", comparison_path.stem)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "comparison_data.csv", rows)
    _write_markdown(output_root / "comparison_table.md", rows)
    print(f"Comparison saved to: {output_root}")


def _load_comparison(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "experiments" not in payload or not isinstance(payload["experiments"], list):
        raise ValueError("Comparison YAML must contain an `experiments` list")
    for item in payload["experiments"]:
        if "name" not in item or "config" not in item:
            raise ValueError("Each comparison experiment needs `name` and `config`")
    return payload


def _summarize_run(name: str, config_path: Path, run_dir: Path, config) -> Dict[str, object]:
    metrics = _read_last_metrics(run_dir / "metrics.jsonl")
    return {
        "name": name,
        "task": config.task.name,
        "trainer": config.trainer.name,
        "pe": _pe_label(config),
        "seed": config.seed,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "final_loss": metrics.get("loss"),
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_answer_accuracy": metrics.get("eval_answer_accuracy"),
        "eval_token_accuracy": metrics.get("eval_token_accuracy"),
        "n_points": metrics.get("n_points"),
    }


def _pe_label(config) -> str:
    if config.model.use_rope:
        return "rope"
    if config.model.use_wpe:
        return f"wpe_{config.model.wpe_mode}"
    return "nope"


def _read_last_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                import json

                last = json.loads(line)
    return last


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    columns = ["name", "task", "trainer", "pe", "seed", "eval_accuracy", "final_loss", "run_dir"]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()

