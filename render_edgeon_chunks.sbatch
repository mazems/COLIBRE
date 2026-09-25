#!/bin/bash
#SBATCH --job-name=relics_edgeon
#SBATCH --array=0-17%6
#SBATCH --output=logs/relics_edgeon_%A_%a.out
#SBATCH --error=logs/relics_edgeon_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swift-conda

cd ~/COLIBRE-analysis || exit 1

CHUNK=15
OFFSET=0

START=$((OFFSET + SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

if [ "$END" -gt 267 ]; then
    END=267
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: rendering galaxies ${START} to $((END-1))"

python images_SRGs.py \
    --snap 127 \
    --orientation edge_on \
    --parallelize \
    --start ${START} \
    --end ${END}
