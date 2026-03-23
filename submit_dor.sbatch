#!/bin/bash
#SBATCH --job-name=DoR
#SBATCH --output=logs/DoR_%A_%a.out
#SBATCH --error=logs/DoR_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=06:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1

TOTAL=129850
CHUNK=1300

START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

source ~/.bashrc
conda activate swift-conda

python3 DoR_computation_noswift.py \
    --compute-all \
    --start $START \
    --end $END \
    --out-csv sfh_times_${SLURM_ARRAY_TASK_ID}.csv.gz
