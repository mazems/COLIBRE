#!/bin/bash
set -euo pipefail

OUTDIR="logs_missing_chunks"
mkdir -p "$OUTDIR"

tasks=(
"270 275 relicness_270_275.csv"
"405 410 relicness_405_410.csv"
"430 435 relicness_430_435.csv"
"1505 1510 relicness_1505_1510.csv"
"1905 1910 relicness_1905_1910.csv"
)

for t in "${tasks[@]}"; do
  read -r a b fn <<<"$t"
  echo
  echo "=== Starting chunk $a $b -> $fn ==="
  jid=$(sbatch -p main -t 08:00:00 --mem=60G --cpus-per-task=1 \
       --job-name=relic_${a}_${b} \
       --output=${OUTDIR}/job_%j_${a}_${b}.out \
       --error=${OUTDIR}/job_%j_${a}_${b}.err \
       --wrap="./run_python.sh ${a} ${b}" | awk '{print $NF}')
  echo "Submitted job $jid — waiting for completion..."

  # wait until job id is no longer in squeue
  while squeue -u $(whoami) -o "%.18i" | awk '{print $1}' | grep -q "^${jid}$"; do
    sleep 10
  done

  echo "Job ${jid} left the queue — printing summary and last lines of log:"
  scontrol show job "${jid}" | sed -n '1,120p'
  echo "---- STDOUT (last 200 lines) ----"
  tail -n 200 ${OUTDIR}/job_${jid}_*.out 2>/dev/null || true
  echo "---- STDERR (last 200 lines) ----"
  tail -n 200 ${OUTDIR}/job_${jid}_*.err 2>/dev/null || true

  if [ -f "${fn}" ]; then
    echo "OK: produced ${fn}"
  else
    echo "WARNING: ${fn} NOT produced. Inspect the .err above."
    # Optionally break so you can debug
    # break
  fi
done

echo "All done."
