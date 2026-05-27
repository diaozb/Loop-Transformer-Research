#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def try_import_yaml():
    try:
        import yaml
        return yaml
    except Exception:
        return None


def flatten_dict(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def load_config(config_path: Path):
    if not config_path.exists():
        return {}

    yaml = try_import_yaml()
    text = config_path.read_text(errors="ignore")

    if yaml is not None:
        try:
            obj = yaml.safe_load(text)
            if isinstance(obj, dict):
                return flatten_dict(obj)
        except Exception:
            pass

    # Fallback: very simple line parser
    flat = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        flat[k.strip()] = v.strip()
    return flat


def get_first_matching_config_value(flat_cfg, names):
    if not flat_cfg:
        return None

    lower_map = {k.lower(): v for k, v in flat_cfg.items()}

    # exact / suffix match
    for name in names:
        name = name.lower()
        for k, v in lower_map.items():
            if k == name or k.endswith("." + name):
                return v

    # loose contains match
    for name in names:
        name = name.lower()
        for k, v in lower_map.items():
            if name in k:
                return v

    return None


def parse_scalar(v):
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = str(v).strip().strip("'").strip('"')
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", s, flags=re.I):
            return float(s)
    except Exception:
        pass
    return s


def infer_weight_bits(quant_tag):
    if quant_tag is None:
        return None
    q = str(quant_tag).lower()
    if q == "fp32" or "fp32" in q:
        return 32
    m = re.search(r"wbits(\d+)", q)
    if m:
        return int(m.group(1))
    return None


def infer_meta_from_path(csv_path: Path, root: Path):
    meta = {
        "setting": None,
        "pe": None,
        "quant_tag": None,
        "run_id": None,
    }

    try:
        rel = csv_path.resolve().relative_to(root.resolve())
        parts = list(rel.parts)
    except Exception:
        parts = list(csv_path.parts)

    if "diagnostics" in parts:
        i = parts.index("diagnostics")
        if i >= 1:
            meta["run_id"] = parts[i - 1]
        if i >= 2:
            meta["quant_tag"] = parts[i - 2]
        if i >= 3:
            meta["pe"] = parts[i - 3]
        if i >= 4:
            meta["setting"] = parts[i - 4]
    else:
        # Expected fallback:
        # .../<setting>/<pe>/<quant>/<run_id>/diagnostics_summary.csv
        if len(parts) >= 5:
            meta["run_id"] = parts[-2]
            meta["quant_tag"] = parts[-3]
            meta["pe"] = parts[-4]
            meta["setting"] = parts[-5]

    return meta


def add_alias_columns(df):
    aliases = {
        "auto_acc_answer": [
            "auto_acc_answer",
            "auto_answer_acc",
            "auto_answer_accuracy",
        ],
        "auto_acc_code": [
            "auto_acc_code",
            "auto_code_acc",
            "auto_code_accuracy",
        ],
        "best_forced_acc_answer": [
            "best_forced_acc_answer",
            "best_forced_answer_acc",
            "best_answer_acc",
        ],
        "best_forced_step": [
            "best_forced_step",
            "best_forced_answer_step",
            "best_answer_step",
        ],
        "final_step_acc_answer": [
            "final_step_acc_answer",
            "final_answer_acc",
            "acc_final_step",
        ],
        "expected_exit_step": [
            "expected_exit_step",
            "expected_step",
            "expected_steps",
        ],
        "argmax_exit_step_mean": [
            "argmax_exit_step_mean",
            "argmax_step_mean",
        ],
        "argmax_exit_step_mode": [
            "argmax_exit_step_mode",
            "argmax_step_mode",
        ],
    }

    for target, candidates in aliases.items():
        if target in df.columns:
            continue
        for c in candidates:
            if c in df.columns:
                df[target] = df[c]
                break

    return df


def summarize_one_csv(csv_path: Path, root: Path):
    run_dir = csv_path.parent.parent
    config_path = run_dir / "config.yaml"

    meta = infer_meta_from_path(csv_path, root)
    flat_cfg = load_config(config_path)

    seed = parse_scalar(get_first_matching_config_value(flat_cfg, ["seed"]))
    cfg_weight_bits = parse_scalar(
        get_first_matching_config_value(flat_cfg, ["weight_bits"])
    )
    cfg_pe = parse_scalar(
        get_first_matching_config_value(flat_cfg, ["pe", "positional_encoding"])
    )
    cfg_train_steps = parse_scalar(
        get_first_matching_config_value(
            flat_cfg,
            ["train_steps", "training_steps", "max_steps", "num_steps", "steps"],
        )
    )
    cfg_quant_scope = parse_scalar(
        get_first_matching_config_value(flat_cfg, ["quant_scope", "scope"])
    )
    cfg_exclude_norm = parse_scalar(
        get_first_matching_config_value(flat_cfg, ["exclude_norm"])
    )

    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None

    df = add_alias_columns(df)

    quant_tag = meta["quant_tag"]
    path_weight_bits = infer_weight_bits(quant_tag)
    weight_bits = cfg_weight_bits if cfg_weight_bits is not None else path_weight_bits

    # Add metadata columns
    df.insert(0, "summary_csv", str(csv_path))
    df.insert(0, "run_dir", str(run_dir))
    df.insert(0, "run_id", meta["run_id"])
    df.insert(0, "seed", seed)
    df.insert(0, "train_steps", cfg_train_steps)
    df.insert(0, "exclude_norm", cfg_exclude_norm)
    df.insert(0, "quant_scope", cfg_quant_scope)
    df.insert(0, "weight_bits", weight_bits)
    df.insert(0, "quant_tag", quant_tag)
    df.insert(0, "pe", cfg_pe if cfg_pe is not None else meta["pe"])
    df.insert(0, "setting", meta["setting"])

    if "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        if df["length"].notna().all():
            df["length"] = df["length"].astype(int)

    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower()

    index_row = {
        "setting": meta["setting"],
        "pe": cfg_pe if cfg_pe is not None else meta["pe"],
        "quant_tag": quant_tag,
        "weight_bits": weight_bits,
        "seed": seed,
        "train_steps": cfg_train_steps,
        "quant_scope": cfg_quant_scope,
        "exclude_norm": cfg_exclude_norm,
        "run_id": meta["run_id"],
        "run_dir": str(run_dir),
        "summary_csv": str(csv_path),
        "n_rows": len(df),
    }

    if "split" in df.columns:
        index_row["splits"] = ",".join(sorted(df["split"].dropna().unique()))
    if "length" in df.columns:
        index_row["min_length"] = df["length"].min()
        index_row["max_length"] = df["length"].max()
        index_row["n_lengths"] = df["length"].nunique()

    return df, index_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="../models/precision_ponder",
        help="Root directory containing precision ponder runs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output merged CSV path. Default: <root>/precision_ponder_summary.csv",
    )
    parser.add_argument(
        "--index-out",
        type=str,
        default=None,
        help="Output run index CSV path. Default: <root>/precision_ponder_run_index.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    out_path = Path(args.out) if args.out else root / "precision_ponder_summary.csv"
    index_out_path = (
        Path(args.index_out)
        if args.index_out
        else root / "precision_ponder_run_index.csv"
    )

    csv_paths = sorted(root.rglob("diagnostics_summary.csv"))

    if not csv_paths:
        print(f"[error] No diagnostics_summary.csv found under: {root}")
        return

    all_dfs = []
    index_rows = []
    skipped = []

    for csv_path in csv_paths:
        try:
            df, index_row = summarize_one_csv(csv_path, root)
            if df is None:
                skipped.append((str(csv_path), "empty"))
                continue
            all_dfs.append(df)
            index_rows.append(index_row)
        except Exception as e:
            skipped.append((str(csv_path), repr(e)))

    if not all_dfs:
        print("[error] Found CSV files, but none could be loaded.")
        for p, reason in skipped:
            print(f"  skipped: {p} | {reason}")
        return

    merged = pd.concat(all_dfs, ignore_index=True, sort=False)
    run_index = pd.DataFrame(index_rows)

    front_cols = [
        "setting",
        "pe",
        "quant_tag",
        "weight_bits",
        "seed",
        "train_steps",
        "quant_scope",
        "exclude_norm",
        "run_id",
        "run_dir",
        "summary_csv",
        "split",
        "length",
    ]
    ordered_cols = [c for c in front_cols if c in merged.columns] + [
        c for c in merged.columns if c not in front_cols
    ]
    merged = merged[ordered_cols]

    sort_cols = [
        c
        for c in [
            "setting",
            "pe",
            "weight_bits",
            "quant_tag",
            "seed",
            "run_id",
            "split",
            "length",
        ]
        if c in merged.columns
    ]
    if sort_cols:
        merged = merged.sort_values(sort_cols, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_out_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_path, index=False)
    run_index.to_csv(index_out_path, index=False)

    print(f"[ok] found diagnostics_summary.csv: {len(csv_paths)}")
    print(f"[ok] loaded runs: {len(index_rows)}")
    print(f"[ok] merged rows: {len(merged)}")
    print(f"[ok] wrote summary: {out_path}")
    print(f"[ok] wrote run index: {index_out_path}")

    if skipped:
        print("\n[warn] skipped files:")
        for p, reason in skipped:
            print(f"  {p} | {reason}")

    # Print run counts
    group_cols = [
        c
        for c in ["setting", "pe", "quant_tag", "weight_bits", "seed"]
        if c in run_index.columns
    ]
    if group_cols:
        print("\n=== runs per config ===")
        print(
            run_index.groupby(group_cols, dropna=False)
            .size()
            .rename("n_runs")
            .reset_index()
            .to_string(index=False)
        )

    # Print quick metric pivots
    if "length" in merged.columns:
        keep_lengths = [20, 21, 22, 40, 60]
        d = merged[merged["length"].isin(keep_lengths)].copy()

        metric_list = [
            "auto_acc_answer",
            "best_forced_acc_answer",
            "best_forced_step",
            "expected_exit_step",
            "argmax_exit_step_mean",
            "argmax_exit_step_mode",
            "final_step_acc_answer",
        ]

        pivot_index = [
            c for c in ["pe", "quant_tag", "weight_bits", "seed"] if c in d.columns
        ]

        if len(d) > 0 and pivot_index:
            for metric in metric_list:
                if metric not in d.columns:
                    continue
                print(f"\n=== {metric} by length ===")
                table = d.pivot_table(
                    index=pivot_index,
                    columns="length",
                    values=metric,
                    aggfunc="mean",
                )
                print(table.to_string())


if __name__ == "__main__":
    main()
