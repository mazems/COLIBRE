#!/bin/bash
set -euo pipefail

mkdir -p logs_missing_chunks

batch_size=12   # how many concurrent sbatch invocations to start quickly (not the scheduler concurrency)
sleep_between=0.05

i=0
while read a b fn; do
  echo "Submitting chunk $a $b -> $fn"
  sbatch -p main -t 04:00:00 --mem=60G --cpus-per-task=1 \
    --job-name=relic_${a}_${b} \
    --output=logs_missing_chunks/job_%j_${a}_${b}.out \
    --error=logs_missing_chunks/job_%j_${a}_${b}.err \
    --wrap="./run_python.sh ${a} ${b}"
  i=$((i+1))
  # a tiny sleep helps avoid flooding the scheduler
  sleep ${sleep_between}
done < missing_files.txt

echo "Submitted $i jobs."
