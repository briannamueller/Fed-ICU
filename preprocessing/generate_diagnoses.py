#!/usr/bin/env python3
"""Generate TPC-format diagnosis CSVs with a configurable time window.

Reads raw eICU CSVs (diagnosis.csv, pastHistory.csv, admissionDx.csv)
and produces per-split diagnosis files using the same hierarchical coding
and prevalence filtering as TPC's original ``diagnoses.py``.

The output files land alongside the existing ``diagnoses.csv`` in each
TPC split directory, named ``diagnoses_{window}h.csv``.

Usage:
    python generate_tpc_diagnoses.py --window 24
    python generate_tpc_diagnoses.py --window 12 --tpc-data-dir /path/to/eICU_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hierarchical coding functions (adapted from TPC eICU_preprocessing/diagnoses.py)
# ---------------------------------------------------------------------------

def _add_codes(splits, codes_dict, words_dict, count):
    """Assign hierarchical integer codes to a split diagnosis string."""
    codes = []
    levels = len(splits)

    def _get_or_create(parent_dict, key, depth_prefix):
        nonlocal count
        if key in parent_dict:
            parent_dict[key][2] += 1
            return parent_dict[key][0]
        parent_dict[key] = [count, {}, 0]
        words_dict[count] = depth_prefix
        c = count
        count += 1
        return c

    if levels >= 1:
        codes.append(_get_or_create(codes_dict, splits[0], splits[0]))
    if levels >= 2:
        codes.append(_get_or_create(
            codes_dict[splits[0]][1], splits[1],
            '|'.join(splits[:2])))
    if levels >= 3:
        codes.append(_get_or_create(
            codes_dict[splits[0]][1][splits[1]][1], splits[2],
            '|'.join(splits[:3])))
    if levels >= 4:
        codes.append(_get_or_create(
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1], splits[3],
            '|'.join(splits[:4])))
    if levels >= 5:
        codes.append(_get_or_create(
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1], splits[4],
            '|'.join(splits[:5])))
    if levels >= 6:
        codes.append(_get_or_create(
            codes_dict[splits[0]][1][splits[1]][1][splits[2]][1][splits[3]][1][splits[4]][1], splits[5],
            '|'.join(splits[:6])))
    return codes, count


def _get_mapping_dict(unique_diagnoses):
    """Build hierarchical code mapping from unique diagnosis strings."""
    main_diagnoses = sorted(a for a in unique_diagnoses
                            if not (a.startswith('notes') or a.startswith('admission')))
    adm_diagnoses = sorted(a for a in unique_diagnoses if a.startswith('admission diagnosis'))
    pasthistory_organsystems = sorted(a for a in unique_diagnoses
                                      if a.startswith('notes/Progress Notes/Past History/Organ Systems/'))
    pasthistory_comments = sorted(a for a in unique_diagnoses
                                   if a.startswith('notes/Progress Notes/Past History/Past History Obtain Options'))

    mapping_dict = {}
    codes_dict = {}
    words_dict = {}
    count = 0

    for diagnosis in main_diagnoses:
        splits = diagnosis.split('|')
        codes, count = _add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in adm_diagnoses:
        shortened = diagnosis.replace('admission diagnosis|', '')
        shortened = shortened.replace('All Diagnosis|', '')
        shortened = shortened.replace('Additional APACHE  Information|', '')
        splits = shortened.split('|')
        codes, count = _add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_organsystems:
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Organ Systems/', '')
        splits = shortened.split('/')
        codes, count = _add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    for diagnosis in pasthistory_comments:
        shortened = diagnosis.replace('notes/Progress Notes/Past History/Past History Obtain Options/', '')
        splits = shortened.split('/')
        codes, count = _add_codes(splits, codes_dict, words_dict, count)
        mapping_dict[diagnosis] = codes

    return codes_dict, mapping_dict, count, words_dict


def _find_pointless_codes(diag_dict):
    """Find codes that are parents of only one child (redundant hierarchy)."""
    pointless_codes = []
    for key, value in diag_dict.items():
        if value[2] == 1:
            pointless_codes.append(value[0])
        for next_key, next_value in value[1].items():
            if key.lower() == next_key.lower():
                pointless_codes.append(next_value[0])
        pointless_codes += _find_pointless_codes(value[1])
    return pointless_codes


def _find_rare_codes(cut_off, sparse_df):
    """Find codes with prevalence below cut_off."""
    prevalence = sparse_df.sum(axis=0)
    return list(prevalence.loc[prevalence <= cut_off].index)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TPC-format diagnosis CSVs with a configurable time window.")
    parser.add_argument("--window", type=int, default=24,
                        help="Diagnosis inclusion window in hours (default: 24).")
    parser.add_argument("--tpc-data-dir", required=True,
                        help="Root of TPC-LoS preprocessed data (contains train/val/test subdirs).")
    parser.add_argument("--eicu-raw-dir", required=True,
                        help="Path to raw eICU CSVs (diagnosis.csv, pastHistory.csv, admissionDx.csv).")
    parser.add_argument("--cut-off-prevalence", type=float, default=0.01,
                        help="Minimum prevalence to keep a diagnosis code (default: 0.01 = 1%%).")
    return parser.parse_args()


def load_raw_diagnoses(eicu_raw_dir: Path, window_minutes: int) -> pd.DataFrame:
    """Load and filter raw eICU diagnosis strings within the time window.

    Mimics the SQL in diagnoses.sql but reads from CSVs and uses a
    configurable offset threshold.

    Returns:
        DataFrame with columns [patientunitstayid, diagnosisstring].
    """
    parts = []

    # 1. Current diagnoses
    print("[diag] Loading diagnosis.csv...")
    diag = pd.read_csv(
        eicu_raw_dir / "diagnosis.csv",
        usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"],
    )
    parts.append(
        diag.loc[diag["diagnosisoffset"] < window_minutes, ["patientunitstayid", "diagnosisstring"]]
    )
    print(f"  diagnosis.csv: {len(parts[-1]):,} rows (of {len(diag):,} total)")

    # 2. Past medical history
    print("[diag] Loading pastHistory.csv...")
    ph = pd.read_csv(
        eicu_raw_dir / "pastHistory.csv",
        usecols=["patientunitstayid", "pasthistoryoffset", "pasthistorypath"],
    )
    ph_filtered = ph.loc[ph["pasthistoryoffset"] < window_minutes].copy()
    ph_filtered = ph_filtered.rename(columns={"pasthistorypath": "diagnosisstring"})
    parts.append(ph_filtered[["patientunitstayid", "diagnosisstring"]])
    print(f"  pastHistory.csv: {len(parts[-1]):,} rows (of {len(ph):,} total)")

    # 3. Admission diagnoses
    print("[diag] Loading admissionDx.csv...")
    adm = pd.read_csv(
        eicu_raw_dir / "admissionDx.csv",
        usecols=["patientunitstayid", "admitdxenteredoffset", "admitdxpath"],
    )
    adm_filtered = adm.loc[adm["admitdxenteredoffset"] < window_minutes].copy()
    adm_filtered = adm_filtered.rename(columns={"admitdxpath": "diagnosisstring"})
    parts.append(adm_filtered[["patientunitstayid", "diagnosisstring"]])
    print(f"  admissionDx.csv: {len(parts[-1]):,} rows (of {len(adm):,} total)")

    combined = pd.concat(parts, ignore_index=True).drop_duplicates()
    print(f"[diag] Total: {len(combined):,} unique (patient, diagnosis) rows "
          f"(window < {window_minutes} min = {window_minutes // 60}h)")
    return combined


def build_sparse_diagnoses(
    raw_diag: pd.DataFrame,
    cut_off_prevalence: float,
    valid_patients: set | None = None,
) -> pd.DataFrame:
    """Build sparse diagnosis DataFrame using TPC's hierarchical coding.

    Uses scipy sparse matrices internally to avoid OOM on large patient sets.
    If *valid_patients* is provided, only those patients are included.
    """
    from scipy import sparse as sp

    # Filter to valid patients early to reduce memory
    if valid_patients is not None:
        raw_diag = raw_diag[raw_diag["patientunitstayid"].isin(valid_patients)]
        print(f"[diag] Filtered to {raw_diag['patientunitstayid'].nunique():,} valid patients")

    raw_diag = raw_diag.set_index("patientunitstayid")
    unique_diagnoses = raw_diag["diagnosisstring"].unique()
    print(f"[diag] {len(unique_diagnoses):,} unique diagnosis strings")

    codes_dict, mapping_dict, count, words_dict = _get_mapping_dict(unique_diagnoses)
    print(f"[diag] {count} hierarchical codes created")

    patients = raw_diag.index.unique()
    patients_to_index = {pid: i for i, pid in enumerate(patients)}

    # Group diagnoses per patient
    print("[diag] Grouping diagnoses per patient...")
    diag_grouped = (
        raw_diag.groupby("patientunitstayid")
        .apply(lambda g: g["diagnosisstring"].tolist())
        .to_dict()
    )

    # Build sparse matrix using COO format (memory efficient)
    print("[diag] Building sparse matrix...")
    rows, cols = [], []
    for pid, diag_list in diag_grouped.items():
        row_idx = patients_to_index[pid]
        code_set = set()
        for diag_str in diag_list:
            for code in mapping_dict.get(diag_str, []):
                code_set.add(code)
        for code in code_set:
            rows.append(row_idx)
            cols.append(code)

    num_patients = len(patients)
    data = np.ones(len(rows), dtype=np.float32)
    sparse_mat = sp.csc_matrix((data, (rows, cols)), shape=(num_patients, count))

    # Identify codes to drop BEFORE converting to dense
    pointless_codes = set(_find_pointless_codes(codes_dict))
    col_sums = np.asarray(sparse_mat.sum(axis=0)).ravel()
    cut_off = round(cut_off_prevalence * num_patients)
    rare_codes = set(i for i in range(count) if col_sums[i] <= cut_off)
    drop_codes = pointless_codes | rare_codes
    keep_cols = sorted(set(range(count)) - drop_codes)

    print(f"[diag] Removing {len(rare_codes)} rare + {len(pointless_codes)} redundant codes, "
          f"keeping {len(keep_cols)}")

    # Slice to kept columns and convert to dense
    sparse_mat = sparse_mat[:, keep_cols]
    dense = sparse_mat.toarray()

    col_names = [words_dict.get(c, str(c)) for c in keep_cols]
    sparse_df = pd.DataFrame(dense, index=patients, columns=col_names)
    sparse_df.rename_axis("patient", inplace=True)
    sparse_df.sort_index(inplace=True)

    print(f"[diag] Final matrix: {sparse_df.shape[0]:,} patients × {sparse_df.shape[1]} diagnosis codes "
          f"(prevalence cutoff: {cut_off_prevalence*100:.1f}%)")
    return sparse_df


def main() -> None:
    args = parse_args()
    window_hours = args.window
    window_minutes = window_hours * 60
    tpc_dir = Path(args.tpc_data_dir)
    eicu_raw_dir = Path(args.eicu_raw_dir)
    output_name = f"diagnoses_{window_hours}h.csv"

    # Check if already generated
    first_split = tpc_dir / "train" / output_name
    if first_split.exists():
        print(f"[diag] {output_name} already exists in {tpc_dir / 'train'}. Skipping.")
        return

    print(f"[diag] Generating diagnosis files with {window_hours}h window...")
    print(f"[diag] TPC data dir: {tpc_dir}")
    print(f"[diag] eICU raw dir: {eicu_raw_dir}")
    print()

    # 0. Collect all valid patient IDs from TPC splits
    valid_patients = set()
    for split in ["train", "val", "test"]:
        stays_path = tpc_dir / split / "stays.txt"
        if stays_path.exists():
            valid_patients.update(int(x) for x in stays_path.read_text().strip().split("\n"))
    print(f"[diag] {len(valid_patients):,} patients across TPC splits")

    # 1. Load raw diagnoses with new window
    raw_diag = load_raw_diagnoses(eicu_raw_dir, window_minutes)
    print()

    # 2. Build global sparse matrix (only for valid patients)
    sparse_df = build_sparse_diagnoses(raw_diag, args.cut_off_prevalence, valid_patients)
    print()

    # 3. Split per train/val/test using existing stays.txt files
    for split in ["train", "val", "test"]:
        split_dir = tpc_dir / split
        stays_path = split_dir / "stays.txt"
        if not stays_path.exists():
            print(f"[diag] Skipping {split}: {stays_path} not found.")
            continue

        patient_ids = [int(x) for x in stays_path.read_text().strip().split("\n")]
        split_df = sparse_df.reindex(patient_ids).fillna(0)

        out_path = split_dir / output_name
        split_df.to_csv(out_path)
        print(f"[diag] Wrote {out_path} ({len(split_df):,} patients × {split_df.shape[1]} codes)")

    print(f"\n[diag] Done. Use --diag-window {window_hours}h in generate_eICU_tpc.py to use these files.")


if __name__ == "__main__":
    main()
