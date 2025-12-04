#!/bin/bash
set -euo pipefail
mkdir -p logs_missing_chunks
if [ ! -s tasks_missing_5_updated.txt ]; then
  echo "tasks_missing_5_updated.txt empty or missing - nothing to submit"; exit 0
fi
echo "Will submit the jobs listed in tasks_missing_5_updated.txt (showing first 50 lines):"
sed -n '1,50p' tasks_missing_5_updated.txt
read -p "Proceed to submit these jobs? (type y to proceed) " ans
if [[ "$ans" != "y" ]]; then echo "Aborting."; exit 0; fi
while read a b fn; do
  echo "Submitting chunk ${a}-${b} -> ${fn}"
  sbatch -p main -t 05:00:00 --mem=60G --cpus-per-task=1 \
    --job-name=relic_${a}_${b} \
    --output=logs_missing_chunks/job_%j_${a}_${b}.out \
    --error=logs_missing_chunks/job_%j_${a}_${b}.err \
    --wrap="./run_python.sh ${a} ${b}"
  sleep 0.05
done < tasks_missing_5_updated.txt
echo "Done submitting."
