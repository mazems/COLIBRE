#!/bin/bash
#SBATCH --job-name=relics
#SBATCH --array=15-266
#SBATCH --output=logs/relics_%A_%a.out
#SBATCH --error=logs/relics_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swift-conda

cd ~/COLIBRE-analysis || exit 1

python images_SRGs.py \
    --snap 127 \
    --orientation face_on \
    --parallelize \
    --index ${SLURM_ARRAY_TASK_ID}
