"""Client selection and splitting for federated learning experiments.

Reads a materialized dataset (manifest.json + per-hospital .npz files
written by generate_clients.py) and returns per-client train/test data
ready for FL training.

Materialization is expensive (~7 min on full eICU) and deterministic.
Selection is cheap (seconds) and parameterized — change num_clients,
min_size, sort_mode, train_ratio, or seed without re-materializing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_manifest(partition_dir):
    """Load and return the manifest dict from a materialized partition."""
    manifest_path = Path(partition_dir) / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def _load_hospital(npz_path):
    """Load a single hospital .npz and return its arrays."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "x_ts": data["x_ts"],         # object array of (T_i, F) float32
        "x_static": data["x_static"], # (N, S) float32
        "y": data["y"],               # (N,) int64 or (N,) object (rolling)
        "patient_ids": data["patient_ids"],  # (N,) int64
    }


def _summary_labels(y):
    """Return a 1-D int64 array suitable for stratification.

    For single_horizon, y is already (N,) int64.
    For rolling, y is (N,) object of per-patient arrays — take the max.
    """
    if y.dtype == object:
        return np.array([int(yi.max()) if len(yi) > 0 else 0 for yi in y],
                        dtype=np.int64)
    return y.astype(np.int64)


def _split_hospital(data, train_ratio, seed, rng):
    """Stratified train/test split for one hospital's data."""
    summary = _summary_labels(data["y"])
    rs = int(rng.integers(0, np.iinfo(np.int32).max))
    tr_idx, te_idx = train_test_split(
        np.arange(len(summary)),
        train_size=train_ratio,
        stratify=summary,
        random_state=rs,
    )
    return (
        {k: v[tr_idx] for k, v in data.items()},
        {k: v[te_idx] for k, v in data.items()},
    )


def _kfold_hospital(data, n_splits, seed):
    """Stratified K-fold split for one hospital's data."""
    summary = _summary_labels(data["y"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in skf.split(np.arange(len(summary)), summary):
        folds.append((
            {k: v[tr_idx] for k, v in data.items()},
            {k: v[te_idx] for k, v in data.items()},
        ))
    return folds


def select_clients(
    partition_dir,
    num_clients: int = 0,
    sort_mode: str = "size",
    min_size: int = 10,
    min_prev: float = 0.0,
    min_minority: int = 0,
    train_ratio: float = 0.75,
    seed: int = 1,
    outer_kfold: Optional[int] = None,
) -> dict:
    """Select hospitals from a materialized partition and split for FL.

    Args:
        partition_dir: path to directory containing manifest.json + hospitals/.
        num_clients: number of top hospitals to keep (0 = all qualifying).
        sort_mode: how to rank hospitals for top-k — "size" (patient count),
            "positives" (positive-class count), or "prevalence" (positive rate).
        min_size: minimum patients per hospital.
        min_prev: minimum positive-class prevalence per hospital.
        min_minority: minimum minority-class samples per hospital.
        train_ratio: fraction of data used for training (per hospital).
        seed: random seed for train/test splitting.
        outer_kfold: if set, use K-fold CV instead of single train/test split.

    Returns:
        dict with keys:
            "clients": list of client dicts, each containing:
                "hospital_id": int
                "train"/"test": dicts with x_ts, x_static, y, patient_ids
                  (or "folds": list of (train_dict, test_dict) if outer_kfold)
            "metadata": manifest metadata + selection parameters
    """
    partition_dir = Path(partition_dir)
    manifest = load_manifest(partition_dir)
    hospitals = manifest["hospitals"]

    # ── Filter ──
    candidates = []
    for hid_str, info in hospitals.items():
        n = info["n_patients"]
        if n < max(min_size, 2):
            continue
        counts = info["label_counts"]
        if len(counts) < 2:
            continue
        min_cls = min(int(v) for v in counts.values())
        if min_minority > 0 and min_cls < min_minority:
            continue
        if outer_kfold and min_cls < outer_kfold:
            continue
        if min_prev > 0 and info["prevalence"] < min_prev:
            continue
        candidates.append((hid_str, info))

    if not candidates:
        raise RuntimeError("No hospitals passed the filter criteria.")

    # ── Score and rank ──
    def _score(info):
        if sort_mode == "size":
            return info["n_patients"]
        elif sort_mode == "positives":
            return sum(int(v) for v in info["label_counts"].values()) - min(
                int(v) for v in info["label_counts"].values())
            # ^ total - min_class = positive count for binary
        elif sort_mode == "prevalence":
            return info["prevalence"]
        return info["n_patients"]

    candidates.sort(key=lambda x: _score(x[1]), reverse=True)
    if num_clients > 0 and num_clients < len(candidates):
        candidates = candidates[:num_clients]

    # ── Load and split ──
    rng = np.random.default_rng(seed)
    clients = []
    for hid_str, info in candidates:
        npz_path = partition_dir / f"hospitals/hospital_{hid_str}.npz"
        data = _load_hospital(npz_path)
        entry = {"hospital_id": int(hid_str)}

        if outer_kfold:
            entry["folds"] = _kfold_hospital(data, outer_kfold, seed)
        else:
            train_data, test_data = _split_hospital(data, train_ratio, seed, rng)
            entry["train"] = train_data
            entry["test"] = test_data

        clients.append(entry)

    # ── Build result metadata ──
    meta = {k: v for k, v in manifest.items() if k != "hospitals"}
    meta["selection"] = {
        "num_clients": len(clients),
        "sort_mode": sort_mode,
        "min_size": min_size,
        "min_prev": min_prev,
        "min_minority": min_minority,
        "train_ratio": train_ratio,
        "seed": seed,
    }
    if outer_kfold:
        meta["selection"]["outer_kfold"] = outer_kfold
    meta["client_ids"] = [c["hospital_id"] for c in clients]

    return {"clients": clients, "metadata": meta}


def export_cohort(cohort, output_dir, compress=True):
    """Write a cohort to disk as per-client .npz files + config.json.

    Creates:
        output_dir/
            config.json      — metadata + selection params + per-client label counts
            train/0.npz ... train/{N-1}.npz
            test/0.npz  ... test/{N-1}.npz

    Each .npz contains x_ts (object), x_static (float32), y (int64).

    Args:
        cohort: return value of select_clients().
        output_dir: directory to write to (created if needed).
        compress: use np.savez_compressed (default True).
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    save = np.savez_compressed if compress else np.savez
    meta = cohort["metadata"]
    clients = cohort["clients"]
    label_counts = []

    for idx, client in enumerate(clients):
        # Train
        tr = client["train"]
        save(train_dir / f"{idx}.npz",
             x_ts=tr["x_ts"], x_static=tr["x_static"], y=tr["y"])
        # Test
        te = client["test"]
        save(test_dir / f"{idx}.npz",
             x_ts=te["x_ts"], x_static=te["x_static"], y=te["y"])

        # Label counts as [[class_id, count], ...]
        y_all = np.concatenate([tr["y"], te["y"]])
        labels, counts = np.unique(y_all, return_counts=True)
        label_counts.append([[int(l), int(c)] for l, c in zip(labels, counts)])

    config = {
        "task": meta.get("task"),
        "num_clients": len(clients),
        "num_classes": int(meta.get("num_classes", 2)),
        "n_ts_features": int(meta.get("n_ts_features", 0)),
        "n_static_features": int(meta.get("n_static_features", 0)),
        "n_flat_features": int(meta.get("n_flat_features", 0)),
        "n_diag_features": int(meta.get("n_diag_features", 0)),
        "max_seq_len": int(meta.get("max_seq_len", 0)),
        "hospital_ids": meta.get("client_ids", []),
        "selection": meta.get("selection", {}),
        "client_label_counts": label_counts,
    }
    # Custom serialization so client_label_counts gets one client per line
    placeholder = "__LABEL_COUNTS__"
    config["client_label_counts"] = placeholder
    text = json.dumps(config, indent=2)
    counts_lines = ",\n".join(f"    {json.dumps(c)}" for c in label_counts)
    text = text.replace(f'"{placeholder}"', f"[\n{counts_lines}\n  ]")
    with open(output_dir / "config.json", "w") as f:
        f.write(text + "\n")

    print(f"[export] {len(clients)} clients → {output_dir}")
