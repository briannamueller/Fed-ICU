"""End-to-end preprocessing orchestrator for Fed-eICU.

Runs the eICU feature-engineering pipeline, producing intermediate outputs
that generate_partitions.py reads when building FL client partitions.

Usage:
    python preprocess.py --eicu-dir data/demo_raw
    python preprocess.py --eicu-dir data/raw --output-dir data/processed

Defaults write to ./data/processed/. Pass --force to rebuild over an existing
output directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "data" / "processed"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"[preprocess] Stage failed: {' '.join(str(c) for c in cmd)}")


def _write_config(out_dir: Path, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _prepare_output(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and force:
        print(f"[preprocess] --force: removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run_preprocessing(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    _prepare_output(out_dir, args.force)

    print(f"[preprocess] eICU → {out_dir}")

    # Stage 1: extract raw tables to intermediate CSVs
    _run([
        sys.executable, "preprocessing/extract_tables.py",
        "--eicu-dir", str(args.eicu_dir),
        "--output-dir", str(out_dir),
    ])

    # Stage 2: binning → filters → diagnoses → flat/labels → train/test split
    _run([
        sys.executable, "preprocessing/run_preprocessing.py",
        "--data-dir", str(out_dir),
        "--eicu-dir", str(args.eicu_dir),
        "--min-dx-prevalence", str(args.min_dx_prevalence),
        "--within-prev", str(args.within_prev),
        "--cross-prev", str(args.cross_prev),
        "--mask-mode", args.mask_mode,
        "--decay-rate", str(args.decay_rate),
    ])

    _write_config(out_dir, {
        "eicu_dir": str(Path(args.eicu_dir).resolve()),
        "min_dx_prevalence": args.min_dx_prevalence,
        "within_prev": args.within_prev,
        "cross_prev": args.cross_prev,
        "mask_mode": args.mask_mode,
        "decay_rate": args.decay_rate,
    })
    print(f"\n[preprocess] Done. Output: {out_dir}")
    print(f"[preprocess] Next: python generate_partitions.py "
          f"--task <task> --eicu-dir {args.eicu_dir} --preprocessed-dir {out_dir}")


def parse_args() -> argparse.Namespace:
    from utils.dataset_utils import load_config, apply_config_defaults

    parser = argparse.ArgumentParser(
        description="Fed-eICU preprocessing: raw eICU CSVs → feature arrays.")
    parser.add_argument("--config", default=None,
                        help="Path to YAML config file (default: configs.yaml).")
    parser.add_argument("--eicu-dir", default=None,
                        help="Path to raw eICU data directory (contains patient.csv, etc.).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/processed/).")
    parser.add_argument("--force", action="store_true",
                        help="Remove any existing output directory before running.")

    parser.add_argument("--min-dx-prevalence", type=float, default=0.01,
                        help="Minimum diagnosis-code prevalence (0-1). "
                             "Default 0.01 (1%%) matches Rocheteau et al.")
    parser.add_argument("--within-prev", type=float, default=0.25,
                        help="Within-hospital prevalence threshold for the "
                             "double-threshold feature filter (default 0.25).")
    parser.add_argument("--cross-prev", type=float, default=0.70,
                        help="Fraction of hospitals that must pass the "
                             "within-hospital threshold (default 0.70).")
    parser.add_argument("--mask-mode", choices=["binary", "exponential_decay"],
                        default="exponential_decay",
                        help="Mask computation mode (default exponential_decay).")
    parser.add_argument("--decay-rate", type=float, default=4.0 / 3.0,
                        help="Decay rate for exponential_decay mask (default 4/3).")

    # Load YAML defaults (CLI args override)
    pre_args, _ = parser.parse_known_args()
    cfg = load_config(pre_args.config)
    apply_config_defaults(parser, cfg, key_map={
        "eicu_dir": "eicu_dir",
        "output_dir": "output_dir",
    })

    args = parser.parse_args()
    if not args.eicu_dir:
        parser.error("--eicu-dir is required (set in configs.yaml or pass on CLI)")
    return args


def main() -> None:
    args = parse_args()
    run_preprocessing(args)


if __name__ == "__main__":
    main()
