#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 NSTART NEND" >&2
  exit 1
fi
NSTART="$1"
NEND="$2"

echo "=== sbatch_relic_wrapper.sh START ==="
date
echo "Host: $(hostname)   User: $(whoami)"
echo "CWD: $(pwd)"
echo "nproc: $(nproc)"
echo "free -h:"
free -h
echo "ulimit -a:"
ulimit -a

# activate conda (adjust path if different)
if [ -f "/home/mzemsch/miniconda3/etc/profile.d/conda.sh" ]; then
  . "/home/mzemsch/miniconda3/etc/profile.d/conda.sh"
  conda activate swift-conda || { echo "Failed to activate swift-conda" >&2; exit 1; }
else
  echo "Conda activation script not found at /home/mzemsch/miniconda3/etc/profile.d/conda.sh" >&2
fi

# force line-buffered Python output so logs update continuously
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running compute_relicness_quantities.py ${NSTART} ${NEND}"
date
# Use stdbuf so stdout/stderr are line-buffered in the job output files
stdbuf -oL -eL python3 compute_relicness_quantities.py "${NSTART}" "${NEND}" 
EXIT=$?
date
echo "=== sbatch_relic_wrapper.sh END (exit $EXIT) ==="
exit $EXIT
