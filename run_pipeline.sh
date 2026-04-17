#!/usr/bin/env bash
#$ -S /bin/bash
#$ -N fed_eicu_pipeline
#$ -cwd
#$ -q UI-HM,MANSCI,COB
#$ -pe smp 32
#$ -l mem_free=512G
#$ -o logs/pipeline.out
#$ -e logs/pipeline.err

# ---- Configuration ----
# Set EICU_DIR to your raw eICU data directory before submitting,
# or export it as an environment variable beforehand.
EICU_DIR="${EICU_DIR:-data/raw}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
PROCESSED_DIR="${REPO_DIR}/data/processed"
PARTITIONS_DIR="${REPO_DIR}/data/partitions"

# ---- Environment ----
# Conda env activation scripts reference unbound vars (e.g. ADDR2LINE),
# so enable strict-mode AFTER activation.
eval "$(${HOME}/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate pygtest-cu117
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

set -eo pipefail

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

echo "==> [$(date)] Host: $(hostname)  Python: $(which python)"
python --version
python -c "import pyarrow, pandas, numpy, sklearn; \
print('pyarrow', pyarrow.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__)"

# ---- Stage 1: eICU CSVs → preprocessed feature arrays ----
# Stage 1 is cache-aware: skips if outputs are already present.
# Add --force to rebuild from scratch.
echo "==> [$(date)] Running preprocess.py..."
python preprocess.py \
    --eicu-dir "${EICU_DIR}" \
    --output-dir "${PROCESSED_DIR}"

# ---- Stage 2: Per-hospital FL client partitions for each task ----
for TASK in mortality_24h mortality_48h los_3day los_7day; do
    echo "==> [$(date)] Generating partitions for ${TASK}..."
    python generate_partitions.py \
        --task "${TASK}" \
        --eicu-dir "${EICU_DIR}" \
        --preprocessed-dir "${PROCESSED_DIR}" \
        --output-dir "${PARTITIONS_DIR}"
done

echo "==> [$(date)] Pipeline complete."
