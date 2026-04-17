"""Shared utilities for eICU federated learning data generation."""
import json
import numpy as np
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs.yaml"


def load_config(path=None):
    """Load configs.yaml and return as a flat dict.

    Args:
        path: Path to YAML file. Defaults to <repo_root>/configs.yaml.

    Returns:
        dict with config values, or empty dict if file not found.
    """
    path = Path(path) if path else _DEFAULT_CONFIG
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_config_defaults(parser, config, key_map=None):
    """Set argparse defaults from a config dict.

    Args:
        parser: argparse.ArgumentParser instance (before parse_args).
        config: dict from load_config().
        key_map: optional {argparse_dest: config_key} for names that
            differ between CLI and YAML.  Unmapped names are matched
            by converting dashes to underscores.
    """
    key_map = key_map or {}
    # Build reverse map: config_key → argparse dest
    actions = {a.dest: a for a in parser._actions}
    defaults = {}
    for dest, action in actions.items():
        config_key = key_map.get(dest, dest)
        if config_key in config and config[config_key] is not None:
            defaults[dest] = config[config_key]
    parser.set_defaults(**defaults)


def format_client_counts(statistic):
    """Format per-client label counts as [[class_id, count], ...]."""
    formatted = []
    for client in statistic:
        formatted.append([[int(cls), int(cnt)] for cls, cnt in client])
    return formatted


def build_class_prevalence(counts, num_classes):
    """Compute per-client class prevalence fractions."""
    prevalence = {}
    for client_idx, client_counts in enumerate(counts):
        totals = [0] * num_classes
        total = 0
        for cls, cnt in client_counts:
            if 0 <= cls < num_classes:
                totals[cls] = cnt
                total += cnt
        if total > 0:
            prevalence[client_idx] = [cnt / total for cnt in totals]
        else:
            prevalence[client_idx] = [0.0] * num_classes
    return prevalence


def save_file(
    config_path,
    train_path,
    test_path,
    train_data,
    test_data,
    num_clients,
    num_classes,
    statistic,
    *,
    split_config=None,
):
    """Save per-client train/test NPZ files and config.json."""
    config_path = Path(config_path)
    train_dir = Path(train_path)
    test_dir = Path(test_path)
    config = dict(split_config or {})
    if 'num_clients' not in config:
        config['num_clients'] = num_clients
    if 'num_classes' not in config:
        config['num_classes'] = num_classes
    counts = format_client_counts(statistic)
    if 'client_label_counts' not in config:
        config['client_label_counts'] = counts
    if 'class_prevalence' not in config:
        config['class_prevalence'] = build_class_prevalence(counts, num_classes)

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        target = train_dir / f"{idx}.npz"
        with open(target, 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        target = test_dir / f"{idx}.npz"
        with open(target, 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    # Write config with formatted client_label_counts
    placeholder = "__CLIENT_COUNT_PLACEHOLDER__"
    counts_data = config.get('client_label_counts', [])
    config['client_label_counts'] = placeholder
    text = json.dumps(config, indent=2)
    if counts_data:
        lines = [f"    {json.dumps(client)}" for client in counts_data]
        counts_block = "[\n" + ",\n".join(lines) + "\n  ]"
    else:
        counts_block = "[]"
    text = text.replace(f'"client_label_counts": "{placeholder}"', f'"client_label_counts": {counts_block}')
    with open(config_path, 'w') as f:
        f.write(text + "\n")

    print("Finish generating dataset.\n")


# ─────────────────────────────────────────────────────────────────────
# Hospital-level materialization helpers
# ─────────────────────────────────────────────────────────────────────

def save_hospital_npz(path, x_ts, x_static, y, patient_ids):
    """Save one hospital's full (unsplit) data to a compressed .npz.

    Args:
        path: output file path.
        x_ts: list of float32 arrays, each shape (T_i, n_features).
        x_static: list of float32 arrays, each shape (n_static,).
        y: int64 array (single_horizon) or list of per-patient int64 arrays (rolling).
        patient_ids: list/array of int patient IDs.
    """
    # Variable-length sequences → object array
    ts_arr = np.empty(len(x_ts), dtype=object)
    ts_arr[:] = x_ts

    static_arr = np.stack(x_static).astype(np.float32)
    pid_arr = np.asarray(patient_ids, dtype=np.int64)

    if isinstance(y, np.ndarray) and y.dtype != object:
        y_arr = y
    else:
        # Rolling labels: list of variable-length arrays → object array
        y_arr = np.empty(len(y), dtype=object)
        y_arr[:] = list(y)

    np.savez_compressed(path, x_ts=ts_arr, x_static=static_arr,
                        y=y_arr, patient_ids=pid_arr)


def write_manifest(path, manifest_dict):
    """Write manifest.json with readable formatting."""
    with open(path, 'w') as f:
        json.dump(manifest_dict, f, indent=2)
        f.write('\n')
