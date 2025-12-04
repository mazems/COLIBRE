#!/bin/bash
#SBATCH -p main
#SBATCH -t 10:00:00        # request 10 hours (adjust)
#SBATCH --mem=60G
#SBATCH -c 1
#SBATCH -J build_ucmg_map
#SBATCH -o logs/build_mapping_%j.out
#SBATCH -e logs/build_mapping_%j.err

# Activate conda env (adjust path/env name if needed)
MINICONDA_PREFIX="/home/mzemsch/miniconda3"
CONDA_ENV="swift-conda"
. "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# run the mapping script (assumes file build_ucmg_mapping.py exists)
python3 build_ucmg_mapping.py
