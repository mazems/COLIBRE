#!/bin/bash
set -euo pipefail

TASKS="tasks_missing_5_updated.txt"
OUTDIR="logs_missing_chunks"
mkdir -p "$OUTDIR"

if [ ! -s "$TASKS" ]; then
  echo "No tasks in $TASKS"; exit 1
fi

while read a b fn; do
  echo
  echo "=== Next task: $a $b -> $fn ==="
  # submit
  jid=$(sbatch -p main -t 08:00:00 --mem=60G --cpus-per-task=1 \
    --job-name=relic_${a}_${b} \
    --output=${OUTDIR}/job_%j_${a}_${b}.out \
    --error=${OUTDIR}/job_%j_${a}_${b}.err \
    --wrap="./run_python.sh ${a} ${b}" | awk '{print $NF}')
  echo "Submitted job $jid for chunk ${a}-${b}. Waiting for it to start and finish..."

  # wait for job to leave the queue (either COMPLETED or FAILED/CANCELLED)
  while squeue -u $(whoami) -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.9C %R" | awk '{print $1}' | grep -q "^${jid}$"; do
    sleep 10
  done

  # show job info and tail the outputs for quick diagnosis
  echo "Job $jid finished (or removed from queue). scontrol summary:"
  scontrol show job ${jid} | sed -n '1,200p'

  # show last 200 lines of stdout/stderr (if present)
  outf=$(ls ${OUTDIR}/job_${jid}_* 2>/dev/null || true)
  echo "---- stdout/err snippet for job $jid ----"
  tail -n 200 ${OUTDIR}/job_${jid}_*.out 2>/dev/null || true
  tail -n 200 ${OUTDIR}/job_${jid}_*.err 2>/dev/null || true

  # check the expected CSV
  if [ -f "${fn}" ]; then
    echo "Output file ${fn} exists — good. Moving to next task."
  else
    echo "WARNING: ${fn} NOT found. You may want to inspect the .err/.out above and re-run this task manually later."
    # optional: break here so you can inspect before continuing:
    # read -p "Press Enter to continue with next task (or Ctrl-C to stop)..." _ || true
  fi

done < "$TASKS"
echo "All tasks processed."
