#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


TARGETS = [
    ("nope", 32, "nope_copy_ponder_bits32.txt"),
    ("nope", 16, "nope_copy_ponder_bits16.txt"),
    ("nope", 8,  "nope_copy_ponder_bits8.txt"),
    ("rope", 32, "rope_copy_ponder_bits32.txt"),
    ("rope", 16, "rope_copy_ponder_bits16.txt"),
    ("rope", 8,  "rope_copy_ponder_bits8.txt"),
]

QUANT_PATTERNS = {
    32: ["fp32", "bits32*", "wbits32*"],
    16: ["wbits16*", "bits16*"],
    8:  ["wbits8*", "bits8*"],
}


def relpath(p: Path) -> str:
    try:
        return os.path.relpath(str(p), os.getcwd())
    except Exception:
        return str(p)


def unique_paths(paths):
    seen = set()
    out = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def run_dir_from_summary(summary_csv: Path) -> Path:
    if summary_csv.parent.name == "diagnostics":
        return summary_csv.parent.parent
    return summary_csv.parent


def run_id_from_summary(summary_csv: Path) -> str:
    return run_dir_from_summary(summary_csv).name


def read_train_steps(run_dir: Path):
    for name in ["config.yaml", "config.yml", "config.json"]:
        p = run_dir / name
        if not p.exists():
            continue

        text = p.read_text(errors="ignore")
        keys = [
            "train_steps",
            "TRAIN_STEPS",
            "training_steps",
            "max_steps",
            "num_steps",
            "steps",
        ]

        for key in keys:
            m = re.search(rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*["\']?(\d+)', text)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass

    return None


def looks_like_diagnostics_summary(csv_path: Path) -> bool:
    try:
        df = pd.read_csv(csv_path, nrows=3)
    except Exception:
        return False

    if df.empty:
        return False

    cols = set(df.columns)
    return ("length" in cols) and (
        ("split" in cols)
        or ("auto_answer_acc" in cols)
        or ("auto_acc_answer" in cols)
        or ("best_forced_answer_acc" in cols)
        or ("best_forced_acc_answer" in cols)
    )


def find_latest_csv(root: Path, setting: str, pe: str, bits: int, min_train_steps: int):
    candidates = []
    qpatterns = QUANT_PATTERNS[bits]

    # Main expected layout:
    # ../models/precision_ponder/copy_ponder/nope/fp32/<run_id>/diagnostics/diagnostics_summary.csv
    for qpat in qpatterns:
        candidates.extend((root / setting / pe).glob(f"{qpat}/*/diagnostics/diagnostics_summary.csv"))
        candidates.extend((root / setting / pe).glob(f"{qpat}/*/diagnostics_summary.csv"))

    # Fallback if root is one level higher, e.g. ../models
    for qpat in qpatterns:
        candidates.extend(root.glob(f"*/{setting}/{pe}/{qpat}/*/diagnostics/diagnostics_summary.csv"))
        candidates.extend(root.glob(f"*/{setting}/{pe}/{qpat}/*/diagnostics_summary.csv"))

    candidates = unique_paths([p for p in candidates if p.is_file()])

    filtered = []
    for csv_path in candidates:
        run_dir = run_dir_from_summary(csv_path)
        train_steps = read_train_steps(run_dir)

        if min_train_steps > 0 and train_steps is not None and train_steps < min_train_steps:
            continue

        if looks_like_diagnostics_summary(csv_path):
            filtered.append(csv_path)

    if not filtered:
        return None, candidates

    # latest = final diagnostics csv modification time
    filtered.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return filtered[0], filtered


def parse_lengths(s: str):
    if s is None or str(s).strip() == "":
        return None

    out = set()
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.update(range(a, b + 1))
            else:
                out.update(range(b, a + 1))
        else:
            out.add(int(part))

    return sorted(out)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias_pairs = [
        ("auto_acc_answer", "auto_answer_acc"),
        ("auto_answer_accuracy", "auto_answer_acc"),
        ("auto_acc_full", "auto_code_acc"),
        ("auto_acc_code", "auto_code_acc"),
        ("auto_code_accuracy", "auto_code_acc"),
        ("best_forced_acc_answer", "best_forced_answer_acc"),
        ("best_answer_acc", "best_forced_answer_acc"),
        ("best_forced_step", "best_forced_answer_step"),
        ("best_answer_step", "best_forced_answer_step"),
        ("min_step_loss", "min_answer_step_loss"),
        ("min_loss_step", "min_answer_loss_step"),
    ]

    for src, dst in alias_pairs:
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    return df


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sort_cols = []

    if "split" in df.columns:
        split_order = {
            "id": 0,
            "train": 0,
            "val": 0,
            "valid": 0,
            "ood": 1,
            "test": 1,
        }
        df["_split_order"] = df["split"].astype(str).str.lower().map(split_order).fillna(9)
        sort_cols.append("_split_order")

    if "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        sort_cols.append("length")

    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    if "_split_order" in df.columns:
        df = df.drop(columns=["_split_order"])

    if "length" in df.columns and df["length"].notna().all():
        df["length"] = df["length"].astype(int)

    return df


def select_display_columns(df: pd.DataFrame):
    preferred = [
        "split",
        "length",
        "auto_answer_acc",
        "auto_code_acc",
        "expected_exit_step",
        "argmax_exit_step_mean",
        "argmax_exit_step_mode",
        "best_forced_answer_acc",
        "best_forced_answer_step",
        "min_answer_step_loss",
        "min_answer_loss_step",
        "final_step_acc_answer",
    ]

    cols = [c for c in preferred if c in df.columns]

    skip_keywords = ["run_dir", "summary_csv", "run_id", "config", "path"]

    for c in df.columns:
        if c in cols:
            continue
        if any(k in c.lower() for k in skip_keywords):
            continue
        if c.startswith("_"):
            continue
        cols.append(c)

    return cols


def write_one_report(summary_csv: Path, out_path: Path, pe: str, bits: int, lengths, width: int):
    run_dir = run_dir_from_summary(summary_csv)
    run_id = run_id_from_summary(summary_csv)

    df = pd.read_csv(summary_csv)
    df = normalize_columns(df)

    if lengths is not None and "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        df = df[df["length"].isin(lengths)].copy()

    df = sort_rows(df)
    display_cols = select_display_columns(df)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", width,
        "display.max_colwidth", 80,
    ):
        table = df[display_cols].to_string(index=False)

    line = "-" * 120

    text = (
        f"Latest Run: {run_id}\n"
        f"Run Dir: {relpath(run_dir)}\n"
        f"Summary CSV: {relpath(summary_csv)}\n"
        f"{line}\n"
        f"{table}\n"
        f"{line}\n"
    )

    out_path.write_text(text, encoding="utf-8")

    return {
        "pe": pe,
        "bits": bits,
        "file": str(out_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_csv": str(summary_csv),
        "n_rows": len(df),
        "mtime": datetime.fromtimestamp(summary_csv.stat().st_mtime).isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export latest precision PonderNet diagnostics into teacher-friendly txt reports."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="../models/precision_ponder",
        help="Root directory containing precision_ponder runs.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="copy_ponder",
        help="Experiment setting directory name.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../reports/precision_ponder_txts",
        help="Directory to write six txt reports.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="",
        help="Optional length filter, e.g. '1-20,21,22,40,60'. Empty means keep all rows.",
    )
    parser.add_argument(
        "--min-train-steps",
        type=int,
        default=0,
        help="Skip runs whose config train_steps is known and below this value. Use 100000 to ignore debug runs.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=160,
        help="Pandas table display width.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any of the six target reports is missing.",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = parse_lengths(args.lengths)

    if not root.exists():
        print(f"[error] root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    missing = []

    for pe, bits, filename in TARGETS:
        selected, candidates = find_latest_csv(
            root=root,
            setting=args.setting,
            pe=pe,
            bits=bits,
            min_train_steps=args.min_train_steps,
        )

        if selected is None:
            print(f"[missing] {pe} bits{bits}: no diagnostics_summary.csv found")
            missing.append((pe, bits, len(candidates)))
            continue

        out_path = out_dir / filename

        row = write_one_report(
            summary_csv=selected,
            out_path=out_path,
            pe=pe,
            bits=bits,
            lengths=lengths,
            width=args.width,
        )
        row["n_candidates"] = len(candidates)
        rows.append(row)

        print(
            f"[ok] {filename} <- {relpath(selected)} "
            f"(run={row['run_id']}, candidates={len(candidates)})"
        )

    if rows:
        index_df = pd.DataFrame(rows)
        index_path = out_dir / "selected_precision_ponder_reports.csv"
        index_df.to_csv(index_path, index=False)
        print(f"\n[ok] wrote selection index: {relpath(index_path)}")

    if missing:
        print("\n[warn] missing targets:")
        for pe, bits, n in missing:
            print(f"  - {pe} bits{bits} candidates_seen={n}")

        if args.strict:
            sys.exit(2)

    print(f"\n[done] reports are in: {relpath(out_dir)}")


if __name__ == "__main__":
    main()
