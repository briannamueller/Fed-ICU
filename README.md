# Fed-eICU: Benchmarking Federated Learning under Natural Cross-Site Heterogeneity

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Benchmarking federated learning using naturally distributed clinical data from the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/).

---

## Overview

The eICU Collaborative Research Database contains over 200,000 ICU stays from 208 U.S. hospitals, making it one of the few large-scale, publicly available clinical datasets where federated clients can be defined by real institutional boundaries rather than artificial partitions. Most federated learning benchmarks simulate heterogeneity by splitting datasets like MNIST or CIFAR across clients in ways that may not fully capture the heterogeneity encountered in real-world deployments. Fed-eICU instead provides naturally heterogeneous partitions defined by hospital identity, where differences in patient populations, clinical practices, and data availability emerge organically across sites. Fed-eICU is an end-to-end pipeline that takes raw eICU data downloaded from PhysioNet and produces ready-to-use client datasets, with the goal of making real-world heterogeneous benchmarking accessible to the federated learning research community.

---

## Quickstart

Requires Python 3.8+. The demo dataset needs ~3 GB of RAM; the full eICU dataset needs ~64 GB. The repo includes the [eICU demo dataset](https://physionet.org/content/eicu-crd-demo/) so you can run the full pipeline without PhysioNet credentials.

```bash
pip install -r requirements.txt

bash run_pipeline.sh
```

This runs all three stages for every task and produces ready-to-use cohorts in `data/cohorts/`. The individual stages are:

1. **`preprocess.py`** -- raw eICU CSVs → cleaned feature arrays in `data/processed/`
2. **`generate_partitions.py --task <task>`** -- groups patients by hospital, writes one `.npz` per hospital + `partition.json` to `data/partitions/<task>/`
3. **`select_cohort.py --task <task>`** -- selects and ranks hospitals, splits train/test, exports per-client `.npz` files to `data/cohorts/<task>/`

Stages 1 and 2 are deterministic and cached -- they skip if outputs already exist (pass `--force` to rebuild). Stage 3 is cheap and parameterized; selection parameters (number of clients, ranking, filters) are configured in `configs.yaml` or passed on the CLI.

For direct use in Python:

```python
from utils.client_selector import select_clients

cohort = select_clients(
    "data/partitions/mortality_24h",
    num_clients=20,
    sort_mode="size",
    train_ratio=0.75,
    seed=42,
)
```

### Using full eICU data

The full dataset requires [PhysioNet credentialed access](https://physionet.org/content/eicu-crd/):

1. Create a [PhysioNet](https://physionet.org/) account
2. Complete the required CITI training
3. Request access to the [eICU Collaborative Research Database](https://physionet.org/content/eicu-crd/2.0/)
4. Download and extract the data
5. Set `eicu_dir` in `configs.yaml` or pass `--eicu-dir /path/to/eicu` on the command line.



---

## Pipeline

### Stage 1: Preprocessing (`preprocess.py`)

Raw eICU CSVs → cleaned feature arrays in `data/processed/`.

Hourly timeseries binning, double-threshold prevalence filtering, 5/95 percentile normalization, GRU-D-style exponential decay missingness masks, forward-fill imputation, diagnosis code extraction, demographic one-hot encoding, and 70/15/15 train/val/test splitting.

### Stage 2: Partitioning (`generate_partitions.py`)

Groups patients by hospital and saves one `.npz` per hospital plus a `partition.json` with per-hospital metadata (counts, label distributions, prevalence).

Each `.npz` contains:
- `x_ts` -- variable-length timeseries per patient (object array of float32)
- `x_static` -- flat + diagnosis features (float32)
- `y` -- labels (int64)
- `patient_ids` -- for traceability (int64)

Example `partition.json`:

```json
{
  "task": "mortality_24h",
  "num_classes": 2,
  "max_seq_len": 24,
  "n_ts_features": 108,
  "n_static_features": 355,
  "n_flat_features": 65,
  "n_diag_features": 290,
  "total_hospitals": 196,
  "total_patients": 180765,
  "hospitals": {
    "56": { "n_patients": 252, "label_counts": {"0": 240, "1": 12}, "prevalence": 0.048 },
    "73": { "n_patients": 6594, "label_counts": {"0": 6052, "1": 542}, "prevalence": 0.082 },
    ...
  }
}
```

### Stage 3: Cohort Selection (`select_cohort.py`)

Selects hospitals from a materialized partition, splits train/test per hospital, and exports ready-to-use per-client `.npz` files and a `config.json` to `data/cohorts/<task>/`. Cheap and parameterized -- no re-materialization needed.

Example `config.json`:

```json
{
  "task": "mortality_24h",
  "num_clients": 20,
  "sort_mode": "size",
  "min_size": 10,
  "min_prev": 0.0,
  "min_minority": 0,
  "train_ratio": 0.75,
  "seed": 1,
  "num_classes": 2,
  "max_seq_len": 24,
  "n_ts_features": 108,
  "n_static_features": 355,
  "n_flat_features": 65,
  "n_diag_features": 290,
  "hospital_ids": [73, 264, 167, ...],
  "client_label_counts": [
    [[0, 6052], [1, 542]],
    [[0, 4396], [1, 519]],
    ...
  ]
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 20 | Number of hospitals to select (0 = all) |
| `sort_mode` | `"size"` | Ranking: `"size"`, `"positives"`, or `"prevalence"` |
| `min_size` | 10 | Minimum patients per hospital |
| `min_prev` | 0.0 | Minimum positive-class prevalence |
| `min_minority` | 0 | Minimum minority-class samples |
| `train_ratio` | 0.75 | Train fraction (stratified per hospital) |
| `seed` | 1 | Random seed |
| `outer_kfold` | None | K-fold CV instead of single split |

---

## Tasks

| Task | Observation Window | Description |
|------|--------|-------------|
| `mortality_24h` | 24h | In-hospital mortality |
| `mortality_48h` | 48h | In-hospital mortality |
| `los_3day` | 72h | Length of stay > 3 days |
| `los_7day` | 168h | Length of stay > 7 days |

---

## Configuration

All defaults live in `configs.yaml`. All three pipeline scripts read it automatically; CLI arguments override any value. Pass `--config /path/to/custom.yaml` to use a different file.

---

## Project Structure

```
Fed-eICU/
├── configs.yaml                # Default configuration for all stages
├── requirements.txt            # Python dependencies
├── run_pipeline.sh             # Run all three stages end-to-end
├── preprocess.py               # Stage 1: raw eICU → feature arrays
├── generate_partitions.py      # Stage 2: feature arrays → per-hospital .npz
├── select_cohort.py            # Stage 3: select hospitals → exported cohort
├── preprocessing/              # Feature engineering modules
├── utils/
│   ├── client_selector.py      # select_clients(), export_cohort()
│   └── dataset_utils.py        # Config loading, NPZ helpers
└── data/
    └── demo_raw/               # eICU demo dataset (ODbL licensed)
```

---

## Citation

```bibtex
@misc{fed-eicu,
  title  = {Fed-eICU: Benchmarking Federated Learning under Natural Cross-Site Heterogeneity},
  author = {Mueller, Brianna},
  year   = {2026},
  url    = {https://github.com/briannamueller/Fed-ICU}
}
```

Please also cite the [eICU Collaborative Research Database](https://doi.org/10.1038/sdata.2018.178).

- **eICU-CRD**: Pollard, T. J., Johnson, A. E., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. *Scientific Data*, 5(1), 1-13.

---

## License

- **Code**: [Apache 2.0](LICENSE)
- **Demo data**: [ODbL](data/demo_raw/LICENSE.txt) via [eICU-CRD Demo](https://physionet.org/content/eicu-crd-demo/)
- **Full eICU**: requires [PhysioNet DUA](https://physionet.org/content/eicu-crd/), not included
