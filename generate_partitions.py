#!/usr/bin/env python
"""Materialize per-hospital FL client data from preprocessed eICU output.

Each hospital is saved as an individual .npz file containing all of its
patients' data (unsplit). A partition.json records per-hospital metadata
so that experiment-time client selection (top-k, filtering, train/test
splitting) can happen without re-materializing data.

Usage:
    python generate_partitions.py --task mortality_24h \\
        --eicu-dir data/raw --preprocessed-dir data/processed

Client selection at experiment time:
    from utils.client_selector import select_clients
    result = select_clients("data/partitions/mortality_24h",
                            num_clients=20, sort_mode="positives",
                            train_ratio=0.75, seed=42)
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.dataset_utils import (
    save_hospital_npz, write_partition_meta, load_config, apply_config_defaults,
)


# ─────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────

# Supported tasks: name → max_seq_len in hours
TASKS = {
    "mortality_24h": 24,
    "mortality_48h": 48,
    "los_3day": 72,
    "los_7day": 168,
}


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize per-hospital FL client data from preprocessed eICU output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: configs.yaml).")
    parser.add_argument("--task", required=True,
                        help="Prediction task (e.g. mortality_24h, mortality_48h, los_3day, los_7day).")
    parser.add_argument("--eicu-dir", default=None,
                        help="Path to raw eICU data directory (contains patient.csv, etc.).")
    parser.add_argument("--preprocessed-dir", default=None,
                        help="Root of preprocessed data (contains train/val/test). "
                             "Defaults to data/processed/.")

    # Feature options
    parser.add_argument("--include-diagnoses", action="store_true", default=True,
                        help="Include diagnosis features.")
    parser.add_argument("--drop-hospital-vars", action="store_true", default=False,
                        help="Drop hospital-level flat features.")
    parser.add_argument("--diag-window", type=str, default="5h",
                        help="Diagnosis time window (e.g. 5h, 24h).")
    parser.add_argument("--paradigm", choices=["single_horizon", "rolling"],
                        default="single_horizon",
                        help="Label paradigm. single_horizon: one scalar label per "
                             "patient. rolling: per-timestep labels.")

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Root output directory for FL partitions (default: data/partitions/).")
    parser.add_argument("--partition-id", default="",
                        help="Custom partition directory name. Auto-generated if empty.")

    # Materialization filters (minimal — keeps any hospital usable for future experiments)
    parser.add_argument("--min-size", type=int, default=2,
                        help="Minimum patients per hospital to materialize (default 2).")
    parser.add_argument("--min-minority", type=int, default=1,
                        help="Minimum minority-class samples per hospital (default 1).")

    # Load YAML defaults (CLI args override)
    pre_args, _ = parser.parse_known_args()
    cfg = load_config(pre_args.config)
    apply_config_defaults(parser, cfg, key_map={
        "eicu_dir": "eicu_dir",
        "output_dir": "partitions_dir",
        "preprocessed_dir": "output_dir",
    })

    args = parser.parse_args()
    if not args.eicu_dir:
        parser.error("--eicu-dir is required (set in configs.yaml or pass on CLI)")
    return args


# ─────────────────────────────────────────────────────────────────────
# Partition ID helper
# ─────────────────────────────────────────────────────────────────────

def build_partition_id(args) -> str:
    """Build a descriptive partition directory name."""
    if args.partition_id:
        return args.partition_id
    parts = [args.task]
    if args.paradigm == "rolling":
        parts.append("rolling")
    return "_".join(parts)


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

_HOSPITAL_LEVEL_PREFIXES = ("teachingstatus", "numbedscategory_", "region_", "physicianspeciality_")


def _load_split(split_dir, max_seq_len, include_diagnoses, drop_hospital_vars, diag_window):
    """Load one preprocessed split directory and return per-patient arrays."""
    stays_path = split_dir / "stays.txt"
    patient_ids = [int(x) for x in stays_path.read_text().strip().split("\n")]
    patient_set = set(patient_ids)

    # Labels
    labels_df = pd.read_csv(split_dir / "labels.csv")
    pid_col = "patient" if "patient" in labels_df.columns else "patientunitstayid"
    labels_df = labels_df[labels_df[pid_col].isin(patient_set)].set_index(pid_col)

    # Flat features
    flat_df = pd.read_csv(split_dir / "flat.csv").set_index("patient")
    if drop_hospital_vars:
        drop_cols = [c for c in flat_df.columns
                     if any(c.startswith(p) for p in _HOSPITAL_LEVEL_PREFIXES)]
        if drop_cols:
            flat_df = flat_df.drop(columns=drop_cols)

    # Diagnoses
    diag_df = None
    if include_diagnoses:
        diag_path = split_dir / ("diagnoses.csv" if diag_window == "5h"
                                  else f"diagnoses_{diag_window}.csv")
        if diag_path.exists():
            diag_df = pd.read_csv(diag_path).set_index("patient")
        else:
            fallback = split_dir / "diagnoses.csv"
            if fallback.exists():
                print(f"[warn] {diag_path.name} not found, falling back to diagnoses.csv")
                diag_df = pd.read_csv(fallback).set_index("patient")

    # Timeseries — prefer parquet (~10x faster than chunked CSV), fall back to CSV.
    ts_parquet = split_dir / "timeseries.parquet"
    ts_csv = split_dir / "timeseries.csv"
    if ts_parquet.exists():
        ts_df = pd.read_parquet(ts_parquet)
        if "patient" in ts_df.columns:
            ts_df = ts_df.set_index("patient")
        meta_cols = {"time", "hour"}
        value_cols = [c for c in ts_df.columns if c not in meta_cols and not c.endswith("_mask")]
        mask_cols = [c for c in ts_df.columns if c.endswith("_mask")]
        feature_cols = value_cols + mask_cols

        patient_data = {}
        for pid, group in ts_df.groupby(level=0, sort=False):
            pid = int(pid)
            if pid not in patient_set:
                continue
            patient_data[pid] = group[feature_cols].to_numpy(dtype=np.float32)
        del ts_df
    else:
        ts_peek = pd.read_csv(ts_csv, nrows=0)
        meta_cols = {"patient", "time", "hour"}
        value_cols = [c for c in ts_peek.columns if c not in meta_cols and not c.endswith("_mask")]
        mask_cols = [c for c in ts_peek.columns if c.endswith("_mask")]
        feature_cols = value_cols + mask_cols

        patient_data = {}
        for chunk in pd.read_csv(ts_csv, chunksize=5_000_000):
            for pid, group in chunk.groupby("patient"):
                pid = int(pid)
                if pid not in patient_set:
                    continue
                arr = group[feature_cols].to_numpy(dtype=np.float32)
                patient_data[pid] = (np.concatenate([patient_data[pid], arr])
                                     if pid in patient_data else arr)

    n_flat = len(flat_df.columns)
    n_diag = len(diag_df.columns) if diag_df is not None else 0

    ts_features, static_features, valid_pids = [], [], []
    for pid in patient_ids:
        if pid not in patient_data or pid not in labels_df.index:
            continue
        ts = patient_data[pid][:max_seq_len]

        flat_vec = (flat_df.loc[pid].to_numpy(dtype=np.float32)
                    if pid in flat_df.index else np.zeros(n_flat, dtype=np.float32))
        np.nan_to_num(flat_vec, copy=False)  # reindex() NaN rows → 0
        static_parts = [flat_vec]
        if diag_df is not None:
            diag_vec = (diag_df.loc[pid].to_numpy(dtype=np.float32)
                        if pid in diag_df.index else np.zeros(n_diag, dtype=np.float32))
            np.nan_to_num(diag_vec, copy=False)
            static_parts.append(diag_vec)

        ts_features.append(ts)
        static_features.append(np.concatenate(static_parts).astype(np.float32))
        valid_pids.append(pid)

    return valid_pids, ts_features, static_features, labels_df, n_flat, n_diag


def _make_labels(labels_df, patient_ids, ts_features, task, paradigm):
    """Create labels for mortality or LOS tasks.

    Returns:
        summary: 1D int64 array of per-patient scalar labels (used for filtering
            and stratification regardless of paradigm).
        entries: ndarray matching ``summary`` in single_horizon mode, or a list
            of per-patient 1D int64 arrays of length T_i in rolling mode.
    """
    mort_col = labels_df["actualhospitalmortality"]
    mort_binary = ((mort_col == "EXPIRED").astype(int) if mort_col.dtype == object
                   else mort_col.astype(int))
    base_task = "mortality" if task.startswith("mortality") else task

    n = len(patient_ids)
    summary = np.zeros(n, dtype=np.int64)
    rolling_entries: List[np.ndarray] = [] if paradigm == "rolling" else None

    for i, (pid, ts) in enumerate(zip(patient_ids, ts_features)):
        offset_min = float(labels_df.loc[pid, "unitdischargeoffset"])
        T_i = int(ts.shape[0])

        if base_task == "mortality":
            y = int(mort_binary.loc[pid])
            summary[i] = y
            if paradigm == "rolling":
                rolling_entries.append(np.full(T_i, y, dtype=np.int64))
        elif base_task in ("los_3day", "los_7day"):
            threshold_days = 3 if base_task == "los_3day" else 7
            if paradigm == "single_horizon":
                summary[i] = 1 if offset_min / 1440.0 > threshold_days else 0
            else:
                t_hours = np.arange(1, T_i + 1, dtype=np.float64)
                remaining_days = (offset_min - t_hours * 60.0) / 1440.0
                y_arr = (remaining_days > threshold_days).astype(np.int64)
                rolling_entries.append(y_arr)
                summary[i] = int(y_arr.max()) if T_i > 0 else 0
        else:
            raise ValueError(f"Unknown task: {task}")

    entries = summary if paradigm == "single_horizon" else rolling_entries
    return summary, entries


def load_preprocessed_data(args):
    """Load preprocessed eICU data and return arrays + hospital mapping."""
    repo_root = Path(__file__).resolve().parent
    preprocessed_dir = (Path(args.preprocessed_dir) if args.preprocessed_dir
                        else repo_root / "data" / "processed")
    eicu_raw = Path(args.eicu_dir)
    max_seq_len = TASKS[args.task]

    if not (preprocessed_dir / "train").exists():
        print(f"[error] Preprocessed data not found at {preprocessed_dir}")
        print(f"[error] Run the preprocessing pipeline first:")
        print(f"  python preprocess.py --eicu-dir {eicu_raw}")
        sys.exit(1)

    # Load hospital mapping from patient.csv
    for candidate in [eicu_raw / "data" / "patient.csv",
                      eicu_raw / "data" / "patient.csv.gz",
                      eicu_raw / "patient.csv",
                      eicu_raw / "patient.csv.gz"]:
        if candidate.exists():
            patient_csv = candidate
            break
    else:
        print(f"[error] patient.csv not found in {eicu_raw}")
        sys.exit(1)
    patient_df = pd.read_csv(patient_csv, usecols=["patientunitstayid", "hospitalid"])
    hospital_map = patient_df.set_index("patientunitstayid")["hospitalid"].to_dict()

    # Load all splits
    print("[info] Loading preprocessed data...")
    all_pids, all_ts, all_static, all_labels_dfs = [], [], [], []

    for split in ["train", "val", "test"]:
        split_dir = preprocessed_dir / split
        if not split_dir.exists():
            continue
        print(f"  Loading {split}...")
        pids, ts, static, ldf, n_flat, n_diag = _load_split(
            split_dir, max_seq_len, args.include_diagnoses,
            args.drop_hospital_vars, args.diag_window)
        all_pids.extend(pids)
        all_ts.extend(ts)
        all_static.extend(static)
        all_labels_dfs.append(ldf)
        print(f"  {split}: {len(pids)} patients (n_flat={n_flat}, n_diag={n_diag})")

    labels_df = pd.concat(all_labels_dfs)
    labels_df = labels_df[~labels_df.index.duplicated(keep="first")]

    summary_labels, label_entries = _make_labels(
        labels_df, all_pids, all_ts, args.task, args.paradigm,
    )

    print(f"[info] Total: {len(all_pids)} patients, positive rate: {summary_labels.mean():.3f}")

    # Map to hospitals
    hospital_ids = np.array([hospital_map.get(pid, -1) for pid in all_pids])
    valid_mask = hospital_ids >= 0
    if not valid_mask.all():
        print(f"[warn] Dropping {(~valid_mask).sum()} patients with no hospital mapping.")
        keep_idx = np.flatnonzero(valid_mask)
        all_pids = [all_pids[i] for i in keep_idx]
        all_ts = [all_ts[i] for i in keep_idx]
        all_static = [all_static[i] for i in keep_idx]
        summary_labels = summary_labels[keep_idx]
        if isinstance(label_entries, np.ndarray):
            label_entries = label_entries[keep_idx]
        else:
            label_entries = [label_entries[i] for i in keep_idx]
        hospital_ids = hospital_ids[keep_idx]

    df = pd.DataFrame({
        "pid": all_pids,
        "hospital_id": hospital_ids,
        "label": summary_labels,
    })

    def ts_getter(indices):
        return [all_ts[i] for i in indices]

    def static_getter(indices):
        return [all_static[i] for i in indices]

    if isinstance(label_entries, np.ndarray):
        def label_getter(indices):
            return label_entries[indices]
    else:
        def label_getter(indices):
            return [label_entries[i] for i in indices]

    metadata = {
        "max_seq_len": max_seq_len,
        "n_ts_features": int(all_ts[0].shape[1]),
        "n_static_features": int(all_static[0].shape[0]),
        "n_flat_features": n_flat,
        "n_diag_features": n_diag,
        "include_diagnoses": args.include_diagnoses,
        "diag_window": args.diag_window,
        "paradigm": args.paradigm,
        "data_format": "ts_static",
        "data_source": "eicu",
    }

    return df, ts_getter, static_getter, label_getter, "label", metadata


# ─────────────────────────────────────────────────────────────────────
# Main — materialize all qualifying hospitals
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.task not in TASKS:
        print(f"[error] Unknown task: {args.task}. Choose from: {list(TASKS.keys())}")
        sys.exit(1)

    df, ts_getter, static_getter, label_getter, label_col, metadata = load_preprocessed_data(args)

    num_classes = int(df[label_col].nunique())

    # Output paths
    repo_root = Path(__file__).resolve().parent
    output_root = Path(args.output_dir) if args.output_dir else repo_root / "data" / "partitions"
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    dataset_dir = output_root / build_partition_id(args)

    if (dataset_dir / "partition.json").exists():
        print(f"\nPartition already exists at {dataset_dir}. Skipping.")
        return

    hospitals_dir = dataset_dir / "hospitals"
    hospitals_dir.mkdir(parents=True, exist_ok=True)

    # ── Iterate hospitals, filter, and materialize ──
    hospital_meta = {}
    total_patients = 0
    min_size = max(args.min_size, 2)

    for hid, group in df.groupby("hospital_id"):
        indices = group.index.to_numpy()
        summary = group[label_col].to_numpy(dtype=np.int64)

        if len(indices) < min_size:
            continue
        uniq, counts = np.unique(summary, return_counts=True)
        if len(uniq) < 2:
            continue
        if args.min_minority > 0 and int(counts.min()) < args.min_minority:
            continue

        x_ts = ts_getter(indices)
        x_static = static_getter(indices)
        y = label_getter(indices)
        pids = group["pid"].to_numpy()

        hid_int = int(hid)
        npz_path = hospitals_dir / f"hospital_{hid_int}.npz"
        save_hospital_npz(npz_path, x_ts, x_static, y, pids)

        label_counts = {str(int(cls)): int(cnt) for cls, cnt in zip(uniq, counts)}
        pos_count = sum(cnt for cls, cnt in zip(uniq, counts) if cls > 0)
        hospital_meta[str(hid_int)] = {
            "n_patients": int(len(indices)),
            "label_counts": label_counts,
            "prevalence": round(pos_count / len(indices), 6),
        }
        total_patients += len(indices)

    if not hospital_meta:
        raise RuntimeError("No hospitals satisfied the materialization requirements.")

    # ── Write manifest ──
    manifest = {
        "version": 1,
        "dataset_name": "eICU",
        "task": args.task,
        "partition_id": build_partition_id(args),
        "num_classes": num_classes,
        **metadata,
        "total_hospitals": len(hospital_meta),
        "total_patients": total_patients,
        "materialized_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hospitals": hospital_meta,
    }

    write_partition_meta(dataset_dir / "partition.json", manifest)

    print(f"\nMaterialized {len(hospital_meta)} hospitals ({total_patients} patients) "
          f"to {dataset_dir}")
    print(f"  Use utils.client_selector.select_clients() to load subsets for experiments.")


if __name__ == "__main__":
    main()
