#!/bin/bash
set -euo pipefail

# --- EDITED: the job IDs you reported ---
JOBS=(311685 311686 311687 311688 311689 311690 311691 311692)

# helper: pretty-format seconds -> HhMmSs
format_seconds(){
  local s=$1
  if [ "$s" -le 0 ]; then
    echo "0s"
    return
  fi
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local sec=$((s%60))
  local out=""
  if [ "$h" -gt 0 ]; then out="${out}${h}h"; fi
  if [ "$m" -gt 0 ]; then out="${out}${m}m"; fi
  out="${out}${sec}s"
  echo "$out"
}

ME=$(whoami)
printf "%-10s %-40s %-6s %-6s %-10s %-10s\n" JOBID OUTFILE DONE EXP AVG_s ETA

total_remaining=0
total_est_sec=0

for jid in "${JOBS[@]}"; do
  # get scontrol output for the job
  sctl=$(scontrol show job "$jid" 2>/dev/null || true)
  if [ -z "$sctl" ]; then
    printf "%-10s %-40s %-6s %-6s %-10s %-10s\n" "$jid" "NO_JOB" "-" "-" "-" "-"
    continue
  fi

  # extract StdOut path
  outpath=$(echo "$sctl" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' || true)
  if [ -z "$outpath" ]; then
    outpath="(no StdOut)"
  fi
  outbase=$(basename "$outpath" 2>/dev/null || true)

  # infer expected count from filename relicness_<A>_<B>.csv or job name relic_A_B
  A=0; B=0
  if [[ "$outpath" =~ relicness_([0-9]+)_([0-9]+)\.csv ]]; then
    A=${BASH_REMATCH[1]}; B=${BASH_REMATCH[2]}
  else
    # fall back to JobName in scontrol
    if echo "$sctl" | grep -q "JobName="; then
      jn=$(echo "$sctl" | sed -n 's/.*JobName=\([^ ]*\).*/\1/p')
      if [[ "$jn" =~ relic_([0-9]+)_([0-9]+) ]]; then
        A=${BASH_REMATCH[1]}; B=${BASH_REMATCH[2]}
      fi
    fi
  fi
  expected=$(( B - A ))

  # parse progress lines from the stdout file (if present)
  done_count=0
  avg="NaN"
  eta_str="unknown"
  if [ -f "$outpath" ] && [ -s "$outpath" ]; then
    # grep times printed like: "... done (total_formed=...) in 22.39s"
    times=$(grep -oP '\[.*\] subhalo .* done .* in \K[0-9]+(?:\.[0-9]+)?(?=s)' "$outpath" || true)
    if [ -n "$times" ]; then
      # compute stats: count, average
      read done_count avg <<< $(echo "$times" | awk '{
        for(i=1;i<=NF;i++){a[++n]=$i; s+= $i}
      }
      END{
        if(n==0){print 0,"0"; exit}
        asort(a)
        avg = s/n
        printf("%d %.3f", n, avg)
      }')
      remaining=$(( expected - done_count ))
      if [ "$remaining" -lt 0 ]; then remaining=0; fi
      # eta in sec (remaining * avg)
      eta_sec=$(awk -v r=$remaining -v a=$avg 'BEGIN{printf("%.0f", r*a)}')
      eta_str=$(format_seconds "$eta_sec")
      total_remaining=$(( total_remaining + remaining ))
      total_est_sec=$(( total_est_sec + eta_sec ))
    fi
  fi

  printf "%-10s %-40s %4s %4s %10s %10s\n" "$jid" "$outbase" "$done_count" "$expected" "$avg" "$eta_str"
done

if [ "$total_remaining" -gt 0 ]; then
  total_hr=$(format_seconds "$total_est_sec")
  echo ""
  echo "Total remaining halos (all listed jobs): $total_remaining"
  echo "Estimated total remaining walltime (sum of per-job ETAs): $total_hr"
else
  echo ""
  echo "No remaining halos reported for these jobs (logs may be empty or jobs may have finished)."
fi
