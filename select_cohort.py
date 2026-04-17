"""Select and export a federated learning cohort from materialized partitions.

Reads a materialized partition (from generate_partitions.py), selects and
ranks hospitals, splits train/test per hospital, and writes ready-to-use
per-client .npz files.

Usage:
    # Using configs.yaml defaults
    python select_cohort.py --task mortality_24h

    # Override selection params
    python select_cohort.py --task mortality_24h --num-clients 50 --sort-mode positives

    # Custom output location
    python select_cohort.py --task mortality_24h --output-dir data/experiments/my_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.client_selector import select_clients, export_cohort
from utils.dataset_utils import load_config, apply_config_defaults

_REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and export a federated learning cohort.")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: configs.yaml).")
    parser.add_argument("--task", required=True,
                        help="Prediction task (e.g. mortality_24h).")
    parser.add_argument("--partitions-dir", default=None,
                        help="Root of materialized partitions (default: data/partitions/).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for exported cohort. "
                             "Default: data/cohorts/<task>/.")

    # Selection parameters
    parser.add_argument("--num-clients", type=int, default=20,
                        help="Number of hospitals to select (0 = all qualifying).")
    parser.add_argument("--sort-mode", type=str, default="size",
                        choices=["size", "positives", "prevalence"],
                        help="Hospital ranking criterion.")
    parser.add_argument("--min-size", type=int, default=10,
                        help="Minimum patients per hospital.")
    parser.add_argument("--min-prev", type=float, default=0.0,
                        help="Minimum positive-class prevalence per hospital.")
    parser.add_argument("--min-minority", type=int, default=0,
                        help="Minimum minority-class samples per hospital.")
    parser.add_argument("--train-ratio", type=float, default=0.75,
                        help="Train fraction per hospital.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for train/test splitting.")

    # Load YAML defaults (CLI args override)
    pre_args, _ = parser.parse_known_args()
    cfg = load_config(pre_args.config)
    # Flatten the nested selection section
    selection_cfg = cfg.get("selection", {})
    flat_cfg = {**cfg, **selection_cfg}
    # Don't let top-level output_dir (preprocessing output) clobber --output-dir
    flat_cfg.pop("output_dir", None)
    apply_config_defaults(parser, flat_cfg, key_map={
        "partitions_dir": "partitions_dir",
    })

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve partition directory
    partitions_dir = Path(args.partitions_dir) if args.partitions_dir else _REPO_ROOT / "data" / "partitions"
    partition_dir = partitions_dir / args.task

    if not (partition_dir / "manifest.json").exists():
        sys.exit(
            f"[select_cohort] Partition not found: {partition_dir}\n"
            f"Run: python generate_partitions.py --task {args.task}"
        )

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _REPO_ROOT / "data" / "cohorts" / args.task
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir

    if (output_dir / "config.json").exists():
        print(f"[select_cohort] Cohort already exists at {output_dir}. Skipping.")
        return

    # Select and export
    print(f"[select_cohort] Selecting from {partition_dir}")
    print(f"  num_clients={args.num_clients}, sort_mode={args.sort_mode}, "
          f"min_size={args.min_size}, train_ratio={args.train_ratio}, seed={args.seed}")

    cohort = select_clients(
        partition_dir,
        num_clients=args.num_clients,
        sort_mode=args.sort_mode,
        min_size=args.min_size,
        min_prev=args.min_prev,
        min_minority=args.min_minority,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    export_cohort(cohort, output_dir)

    try:
        rel = output_dir.relative_to(_REPO_ROOT)
    except ValueError:
        rel = output_dir
    print(f"[select_cohort] Done. Cohort at {rel}")


if __name__ == "__main__":
    main()
