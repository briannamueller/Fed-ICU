"""Stage A: hourly binning of eICU timeseries sources.

Produces ``binned_timeseries.parquet`` — one row per (patient, hour) with
raw feature values (NaN where not measured). This is the expensive, cached
output that Stage B (``apply_filters.py``) consumes to produce the final
normalized/masked/filtered ``preprocessed_timeseries.csv``.

Parameter-sensitive transformations (double-threshold prevalence filter,
quantile normalization, mask mode) do NOT live here so researchers can
re-run Stage B with different settings without re-binning.
"""
import argparse
import gc
import os
from itertools import islice

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def reconfigure_timeseries_fast(df, offset_column, feature_column=None, test=False):
    """Convert long-format timeseries to wide-format with (patient, hour) index.

    Instead of using timedelta + resample, we convert offsets directly to
    integer hours via ceiling and group by (patient, hour) upfront. Matches
    pandas ``resample('H', closed='right', label='right')`` but much faster.
    """
    if test:
        df = df.iloc[300000:5000000]

    df['hour'] = np.ceil(df[offset_column] / 60).astype(int)
    df.drop(columns=[offset_column], inplace=True)

    if feature_column is not None:
        df = df.groupby(['patientunitstayid', 'hour', feature_column]).mean().reset_index()
        df = df.pivot_table(
            index=['patientunitstayid', 'hour'],
            columns=feature_column,
            values=df.columns.difference(
                ['patientunitstayid', 'hour', feature_column]
            ).tolist(),
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        df.columns.name = None
    else:
        df = df.groupby(['patientunitstayid', 'hour']).mean()

    df.index.names = ['patient', 'time']
    return df


def gen_patient_chunk(patients, size=1000):
    it = iter(patients)
    chunk = list(islice(it, size))
    while chunk:
        yield chunk
        chunk = list(islice(it, size))


def bin_sources(eICU_path, test=False):
    """Load each raw timeseries source, bin to (patient, hour), return dict."""
    print('==> Loading and binning timeseries sources...')
    read_kw = {'nrows': 500000} if test else {}

    print('==> Lab...')
    ts_lab = pd.read_csv(eICU_path + 'timeserieslab.csv', **read_kw)
    ts_lab = reconfigure_timeseries_fast(ts_lab,
                                         offset_column='labresultoffset',
                                         feature_column='labname',
                                         test=test)
    print(f'    Lab shape: {ts_lab.shape}')
    gc.collect()

    print('==> Respiratory...')
    ts_resp = pd.read_csv(eICU_path + 'timeseriesresp.csv', low_memory=False, **read_kw)
    ts_resp = ts_resp.replace('%', '', regex=True)
    ts_resp['respchartvalue'] = pd.to_numeric(ts_resp['respchartvalue'], errors='coerce')
    ts_resp = ts_resp.loc[ts_resp['respchartvalue'].notnull()]
    ts_resp = reconfigure_timeseries_fast(ts_resp,
                                          offset_column='respchartoffset',
                                          feature_column='respchartvaluelabel',
                                          test=test)
    print(f'    Resp shape: {ts_resp.shape}')
    gc.collect()

    print('==> Nurse...')
    ts_nurse = pd.read_csv(eICU_path + 'timeseriesnurse.csv', **read_kw)
    ts_nurse['nursingchartvalue'] = pd.to_numeric(ts_nurse['nursingchartvalue'], errors='coerce')
    ts_nurse = ts_nurse.loc[ts_nurse['nursingchartvalue'].notnull()]
    ts_nurse = reconfigure_timeseries_fast(ts_nurse,
                                           offset_column='nursingchartoffset',
                                           feature_column='nursingchartcelltypevallabel',
                                           test=test)
    print(f'    Nurse shape: {ts_nurse.shape}')
    gc.collect()

    print('==> Aperiodic...')
    ts_aperiodic = pd.read_csv(eICU_path + 'timeseriesaperiodic.csv', **read_kw)
    ts_aperiodic = reconfigure_timeseries_fast(ts_aperiodic,
                                               offset_column='observationoffset',
                                               test=test)
    print(f'    Aperiodic shape: {ts_aperiodic.shape}')
    gc.collect()

    print('==> Periodic...')
    ts_periodic = pd.read_csv(eICU_path + 'timeseriesperiodic.csv', **read_kw)
    ts_periodic = reconfigure_timeseries_fast(ts_periodic,
                                              offset_column='observationoffset',
                                              test=test)
    print(f'    Periodic shape: {ts_periodic.shape}')
    gc.collect()

    return {
        'lab': ts_lab,
        'resp': ts_resp,
        'nurse': ts_nurse,
        'periodic': ts_periodic,
        'aperiodic': ts_aperiodic,
    }


def write_binned(eICU_path, sources, test=False):
    """Merge per-source binned frames per-patient-chunk and write parquet."""
    out_path = eICU_path + 'binned_timeseries.parquet'
    if os.path.exists(out_path):
        os.remove(out_path)

    # Union of feature columns across all sources — parquet row groups must
    # share a schema, so we reindex each chunk to this column set.
    feature_cols = sorted({c for df in sources.values() for c in df.columns})

    # Index sets for fast per-chunk lookup.
    patient_sets = {k: set(v.index.get_level_values(0).unique()) for k, v in sources.items()}
    patients = sources['periodic'].index.unique(level=0)

    size = 20000
    writer = None
    all_stays = []
    try:
        for i, chunk in enumerate(gen_patient_chunk(patients, size=size), start=1):
            chunk_set = set(chunk)
            parts = []
            for key, df in sources.items():
                valid = list(chunk_set & patient_sets[key])
                if valid:
                    parts.append(df.loc[valid])
            merged = pd.concat(parts, sort=True)
            merged.reset_index(level='time', inplace=True)

            # Normalize to a fixed schema: patient (int64), time (int32), then
            # all feature columns as float32 in sorted order.
            merged = merged.reindex(columns=['time'] + feature_cols)
            merged['time'] = merged['time'].astype(np.int32)
            merged[feature_cols] = merged[feature_cols].astype(np.float32)
            merged = merged.reset_index()  # 'patient' → column
            merged['patient'] = merged['patient'].astype(np.int64)

            all_stays.extend(merged['patient'].unique().tolist())

            if not test:
                table = pa.Table.from_pandas(merged, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema,
                                              compression='snappy')
                writer.write_table(table)

            print(f'==> Wrote chunk {i}: {len(chunk)} patients, running total '
                  f'{min(i * size, len(patients))}/{len(patients)}')
            del merged
    finally:
        if writer is not None:
            writer.close()

    # Deduplicated stay list for downstream steps (diagnoses, flat_and_labels).
    stays = sorted(set(int(s) for s in all_stays))
    with open(eICU_path + 'stays.txt', 'w') as f:
        for s in stays:
            f.write(f'{s}\n')
    print(f'==> Wrote stays.txt with {len(stays)} patients')


def timeseries_main(eICU_path, test=False):
    """Stage A entry point: produce binned_timeseries.parquet + stays.txt.

    Cache-aware: if ``binned_timeseries.parquet`` already exists, skip the
    expensive re-binning. Pass --force to preprocessing.py (which removes the
    whole output dir) to bypass the cache.
    """
    parquet_path = eICU_path + 'binned_timeseries.parquet'
    if not test and os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
        print(f'[cached] {parquet_path} already present, skipping binning.')
        return
    if not test:
        try:
            os.remove(eICU_path + 'binned_timeseries.csv')
        except FileNotFoundError:
            pass
    sources = bin_sources(eICU_path, test=test)
    write_binned(eICU_path, sources, test=test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage A: bin eICU timeseries.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing intermediate CSVs from extract_tables.py.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with limited data.')
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'
    timeseries_main(data_dir, test=args.test)
