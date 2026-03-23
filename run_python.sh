#!/bin/bash
# run_python.sh nstart nend
set -euo pipefail

MINICONDA_PREFIX="/home/mzemsch/miniconda3"       # <--- change if different
CONDA_ENV="swift-conda"                          # <--- change if different

# Source conda.sh then activate
if [ -f "${MINICONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
    . "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found at ${MINICONDA_PREFIX}/etc/profile.d/conda.sh" >&2
    exit 1
fi

conda activate "${CONDA_ENV}" || { echo "Failed to activate conda env ${CONDA_ENV}" >&2; exit 1; }

# mitigate multi-threaded libs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# run script (args passed through)
python3 compute_relicness_quantities.py "$1" "$2"
