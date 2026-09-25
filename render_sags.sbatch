#!/bin/bash
#SBATCH --job-name=sags
#SBATCH --array=0-23%6
#SBATCH --output=logs/sags_%A_%a.out
#SBATCH --error=logs/sags_%A_%a.err
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

if [ "$END" -gt 223 ]; then
    END=223
fi

python images_SRGs.py \
    --csv z0_sags_trackids.csv \
    --snap 127 \
    --orientation edge_on \
    --parallelize \
    --output-dir images_sags \
    --start ${START} \
    --end ${END}
