"""Stage B: parameter-sensitive filters and transformations on binned data.

Reads ``binned_timeseries.parquet`` (produced by timeseries.py) and applies:
  1. Double-threshold prevalence filter (within-hospital × cross-hospital)
  2. Global-quantile (5/95) normalization on the full dataset
  3. Value clipping to [-4, 4]
  4. Mask computation (binary or exponential_decay)
  5. Forward-fill within patients
  6. Length-limit clipping (default 24*14 hours)
  7. Time-of-day feature from flat_features.csv

Writes ``preprocessed_timeseries.csv`` — consumed by split_train_test.py.

This stage is cheap to re-run with different parameter settings because the
expensive binning in Stage A is cached in ``binned_timeseries.parquet``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


LENGTH_LIMIT_HOURS = 24 * 14  # 14-day cap, matches TPC


# ─────────────────────────────────────────────────────────────────────
# Prevalence filter (within-hospital × cross-hospital)
# ─────────────────────────────────────────────────────────────────────

def load_hospital_map(eicu_dir: Path) -> dict:
    """Return patientunitstayid → hospitalid from raw eICU patient.csv."""
    for candidate in [eicu_dir / 'patient.csv',
                      eicu_dir / 'patient.csv.gz']:
        if candidate.exists():
            df = pd.read_csv(candidate, usecols=['patientunitstayid', 'hospitalid'])
            return df.set_index('patientunitstayid')['hospitalid'].to_dict()
    raise FileNotFoundError(f'patient.csv not found under {eicu_dir}')


def apply_prevalence_filter(df: pd.DataFrame, hospital_map: dict,
                            within_threshold: float,
                            cross_threshold: float) -> pd.DataFrame:
    """Drop feature columns failing the double-threshold prevalence filter.

    A feature is kept if, in at least ``cross_threshold`` fraction of
    hospitals, the within-hospital prevalence (fraction of that hospital's
    patients with ≥1 non-null observation) is ≥ ``within_threshold``.
    """
    feature_cols = [c for c in df.columns if c != 'time']
    print(f'==> Prevalence filter ({len(feature_cols)} candidate features, '
          f'within≥{within_threshold}, cross≥{cross_threshold})')

    # Per-patient presence: True if the feature has any non-null value.
    presence = df[feature_cols].notnull().groupby(level='patient').any()

    hospitals = presence.index.to_series().map(hospital_map)
    if hospitals.isnull().any():
        missing = int(hospitals.isnull().sum())
        print(f'  [warn] {missing} patients have no hospital mapping; dropping from filter computation')
        presence = presence[hospitals.notnull()]
        hospitals = hospitals.dropna()

    # Within-hospital prevalence per feature.
    within_prev = presence.groupby(hospitals.values).mean()
    # Fraction of hospitals where within-hospital prevalence ≥ threshold.
    hospitals_pass = (within_prev >= within_threshold).mean(axis=0)
    keep = hospitals_pass[hospitals_pass >= cross_threshold].index.tolist()

    dropped = len(feature_cols) - len(keep)
    print(f'  Kept {len(keep)} / {len(feature_cols)} features ({dropped} dropped)')
    return df[['time'] + keep]


# ─────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────

def normalize_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """5/95 percentile min-max scaling to [-1, 1], then clip to [-4, 4]."""
    feature_cols = [c for c in df.columns if c != 'time']
    print(f'==> Computing global 5/95 quantiles on {len(feature_cols)} features...')
    quantiles = df[feature_cols].quantile([0.05, 0.95])
    mins = quantiles.loc[0.05]
    maxs = quantiles.loc[0.95]
    spread = (maxs - mins).replace(0, np.nan)

    df[feature_cols] = 2 * (df[feature_cols] - mins) / spread - 1
    df[feature_cols] = df[feature_cols].clip(lower=-4, upper=4)
    return df


# ─────────────────────────────────────────────────────────────────────
# Masks (vectorized exponential decay or binary)
# ─────────────────────────────────────────────────────────────────────

def compute_exponential_decay_mask(df: pd.DataFrame, decay_rate: float) -> pd.DataFrame:
    """Vectorized GRU-D-style exponential decay mask across all columns."""
    mask_bool = df.notnull()
    inv_mask = ~mask_bool

    patient_ids = df.index.get_level_values('patient').values
    inv_cumsum = inv_mask.cumsum(axis=0)
    cumsum_at_obs = inv_cumsum.where(mask_bool)
    cumsum_at_obs_filled = cumsum_at_obs.groupby(patient_ids).ffill().fillna(0)
    count_non_measurements = inv_cumsum - cumsum_at_obs_filled

    mask_float = mask_bool.astype(float)
    mask_float[~mask_bool] = np.nan
    mask_ffilled = mask_float.groupby(patient_ids).ffill().fillna(0)

    divisor = (count_non_measurements * decay_rate).clip(lower=1)
    return mask_ffilled / divisor


def compute_mask(df: pd.DataFrame, mode: str, decay_rate: float) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != 'time']
    features = df[feature_cols]
    if mode == 'binary':
        mask = features.notnull().astype(float)
    elif mode == 'exponential_decay':
        mask = compute_exponential_decay_mask(features, decay_rate)
    else:
        raise ValueError(f'Unknown mask mode: {mode!r}')
    mask.columns = [f'{c}_mask' for c in mask.columns]
    return mask


# ─────────────────────────────────────────────────────────────────────
# Forward-fill and length limit
# ─────────────────────────────────────────────────────────────────────

def forward_fill_and_limit(df: pd.DataFrame) -> pd.DataFrame:
    """Clip time > 0, time < LENGTH_LIMIT, forward-fill within patients, zero-fill."""
    df = df[(df['time'] > 0) & (df['time'] < LENGTH_LIMIT_HOURS)]
    patient_ids = df.index.get_level_values('patient').values
    feature_cols = [c for c in df.columns if c != 'time']
    df[feature_cols] = df[feature_cols].groupby(patient_ids).ffill()
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────
# Time-of-day feature
# ─────────────────────────────────────────────────────────────────────

def add_time_of_day(df: pd.DataFrame, flat_path: str) -> pd.DataFrame:
    """Attach a normalized time-of-day feature per row.

    ``flat_features.csv`` already stores the admit hour-of-day as ``hour``
    (computed in extract_tables). We add that to the ICU-relative hour
    offset, mod 24, and map onto a normalized [0, 1] position.
    """
    print('==> Adding time-of-day feature...')
    flat = pd.read_csv(flat_path, usecols=['patientunitstayid', 'hour'])
    hour_map = flat.set_index('patientunitstayid')['hour'].to_dict()
    hour_list = np.linspace(0, 1, 24)

    pid_col = df.index.get_level_values('patient')
    admit_hour = pid_col.to_series().map(hour_map).to_numpy(dtype=np.float64)
    combined = df['time'].to_numpy(dtype=np.float64) + admit_hour
    idx = np.where(np.isnan(combined), 0, combined.astype(np.int64) % 24 - 24)
    tod = np.where(np.isnan(combined), 0.0, hour_list[idx])
    df['hour'] = tod
    return df


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def apply_filters_main(data_dir: str, eicu_dir: str,
                       within_threshold: float, cross_threshold: float,
                       mask_mode: str, decay_rate: float) -> None:
    if not data_dir.endswith('/'):
        data_dir += '/'
    eicu_path = Path(eicu_dir)

    print(f'==> Loading binned_timeseries.parquet...')
    df = pd.read_parquet(data_dir + 'binned_timeseries.parquet')
    df = df.set_index('patient').sort_index()
    print(f'  Loaded {len(df)} rows, {df.shape[1] - 1} feature columns')

    # We need a MultiIndex (patient, time) for groupby-by-level operations
    # later; keep time as a column throughout for simple slicing.
    df.index.name = 'patient'

    hospital_map = load_hospital_map(eicu_path)
    df = apply_prevalence_filter(df, hospital_map, within_threshold, cross_threshold)
    df = normalize_and_clip(df)

    # Build a (patient, time) MultiIndex view for the mask/ffill passes.
    df = df.set_index('time', append=True)
    df.index.names = ['patient', 'time']

    mask = compute_mask(df, mode=mask_mode, decay_rate=decay_rate)
    df = pd.concat([df, mask], axis=1)
    df = df.reset_index(level='time')

    df = forward_fill_and_limit(df)
    df = add_time_of_day(df, data_dir + 'flat_features.csv')

    out_path = data_dir + 'preprocessed_timeseries.csv'
    print(f'==> Writing {out_path}...')
    df.to_csv(out_path)
    print(f'==> Done: {len(df)} rows, {df.shape[1]} columns')

    # Authoritative stays manifest: only patients that survived the time-window
    # clipping in forward_fill_and_limit. The earlier stays.txt written by
    # timeseries.py is from the pre-filter binned data and may include patients
    # whose only rows had time<=0 or time>=LENGTH_LIMIT_HOURS — those won't be
    # in preprocessed_timeseries.csv and would break split_train_test.
    surviving = sorted(int(p) for p in df.index.unique())
    stays_path = data_dir + 'stays.txt'
    print(f'==> Writing {stays_path} with {len(surviving)} surviving patients...')
    with open(stays_path, 'w') as f:
        for p in surviving:
            f.write(f'{p}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage B: apply filters + normalization + masks.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing binned_timeseries.parquet and flat_features.csv.')
    parser.add_argument('--eicu-dir', required=True,
                        help='Raw eICU directory (for hospital mapping).')
    parser.add_argument('--within-prev', type=float, default=0.25,
                        help='Within-hospital prevalence threshold (default 0.25).')
    parser.add_argument('--cross-prev', type=float, default=0.70,
                        help='Cross-hospital coverage threshold (default 0.70).')
    parser.add_argument('--mask-mode', choices=['binary', 'exponential_decay'],
                        default='exponential_decay',
                        help='Mask computation mode (default exponential_decay).')
    parser.add_argument('--decay-rate', type=float, default=4.0 / 3.0,
                        help='Decay rate for exponential_decay mask (default 4/3).')
    args = parser.parse_args()
    apply_filters_main(args.data_dir, args.eicu_dir,
                       args.within_prev, args.cross_prev,
                       args.mask_mode, args.decay_rate)
