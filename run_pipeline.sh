#!/usr/bin/env bash
set -eo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PROCESSED_DIR="${REPO_DIR}/data/processed"
PARTITIONS_DIR="${REPO_DIR}/data/partitions"

cd "${REPO_DIR}"

echo "==> [$(date)] Python: $(which python)"
python --version

# ---- Stage 1: eICU CSVs → preprocessed feature arrays ----
# Skips automatically if outputs already exist. Add --force to rebuild.
echo "==> [$(date)] Running preprocess.py..."
python preprocess.py

# ---- Stage 2: Per-hospital partitions for each task ----
for TASK in mortality_24h mortality_48h los_3day los_7day; do
    echo "==> [$(date)] Generating partitions for ${TASK}..."
    python generate_partitions.py --task "${TASK}"
done

# ---- Stage 3: Select and export cohorts for each task ----
for TASK in mortality_24h mortality_48h los_3day los_7day; do
    echo "==> [$(date)] Selecting cohort for ${TASK}..."
    python select_cohort.py --task "${TASK}"
done

echo "==> [$(date)] Pipeline complete."
