#!/bin/bash
# Main study: three feeders x two DER cases x 10 replications (~25 min total)
cd "$(dirname "$0")"
for f in ieee33 ieee69 ieee123; do
  for c in moderate der_rich; do
    python3 run_experiment.py --feeder $f --case $c --reps 10 > results/log_${f}_${c}.txt 2>&1
  done
done
echo ALL_DONE > results/main_done.txt
