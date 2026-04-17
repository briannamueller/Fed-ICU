"""Orchestrate the TPC preprocessing pipeline (Stage 2).

Runs Stage A (timeseries binning) → Stage B (filters + normalization + masks)
→ diagnoses → flat/labels → train/val/test splitting on the intermediate CSVs
produced by extract_tables.py.

Usage:
    python tpc/run_preprocessing.py --data-dir /path/to/intermediate/csvs \
        --eicu-dir /path/to/eICU
"""
import argparse
import os
import sys

# Allow running as `python tpc/run_preprocessing.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from timeseries import timeseries_main
from apply_filters import apply_filters_main
from diagnoses import diagnoses_main
from flat_and_labels import flat_and_labels_main
from split_train_test import split_train_test


def main(data_dir, eicu_dir,
         within_prev=0.25, cross_prev=0.70,
         mask_mode='exponential_decay', decay_rate=4.0 / 3.0,
         min_dx_prevalence=0.01):
    if not data_dir.endswith('/'):
        data_dir += '/'

    print('==> Removing stale stays.txt if present...')
    try:
        os.remove(data_dir + 'stays.txt')
    except FileNotFoundError:
        pass

    # Stage A: cached binning.
    timeseries_main(data_dir, test=False)

    # Stage B: parameter-sensitive filters/normalization/masks.
    apply_filters_main(
        data_dir=data_dir,
        eicu_dir=eicu_dir,
        within_threshold=within_prev,
        cross_threshold=cross_prev,
        mask_mode=mask_mode,
        decay_rate=decay_rate,
    )

    diagnoses_main(data_dir, min_dx_prevalence)
    flat_and_labels_main(data_dir)
    split_train_test(data_dir, is_test=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run TPC preprocessing pipeline on intermediate CSVs.')
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing intermediate CSVs from extract_tables.py.')
    parser.add_argument('--eicu-dir', required=True,
                        help='Raw eICU directory (for hospital mapping in Stage B).')
    parser.add_argument('--min-dx-prevalence', type=float, default=0.01,
                        help='Minimum diagnosis-code prevalence (default 0.01 = 1%%).')
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
    main(args.data_dir, args.eicu_dir,
         within_prev=args.within_prev, cross_prev=args.cross_prev,
         mask_mode=args.mask_mode, decay_rate=args.decay_rate,
         min_dx_prevalence=args.min_dx_prevalence)
