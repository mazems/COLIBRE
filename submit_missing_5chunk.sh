#!/bin/bash
set -euo pipefail
mkdir -p logs_missing_chunks

PARTITION="main"
TIME="08:00:00"
MEM="60G"
CPUS=1
SLEEP=0.12

i=0
while read a b fn; do
  echo "Submitting chunk $a $b -> $fn"
  sbatch -p ${PARTITION} -t ${TIME} --mem=${MEM} -c ${CPUS} \
    --job-name=relic_${a}_${b} \
    --output=logs_missing_chunks/job_%j_${a}_${b}.out \
    --error=logs_missing_chunks/job_%j_${a}_${b}.err \
    --wrap="./run_python.sh ${a} ${b}"
  i=$((i+1))
  sleep ${SLEEP}
done < missing_files_5chunk.txt

echo "Submitted $i jobs."
