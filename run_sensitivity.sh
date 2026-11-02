#!/bin/bash
# Sensitivity to source strength (Zs) and inverter fault-current limit (kf),
# DER-rich case, 5 replications each, no group-count sweep (~15 min total).
# Base case (Zs = j0.20 pu, kf = 2.0) comes from run_all.sh.
cd "$(dirname "$0")"
for f in ieee33 ieee69 ieee123; do
  for zs in 0.10 0.40; do
    python3 run_experiment.py --feeder $f --case der_rich --reps 5 --no-sweep --zs $zs --tag _zs$zs \
      > results/log_${f}_der_rich_zs$zs.txt 2>&1
  done
  for kf in 1.2 1.5; do
    python3 run_experiment.py --feeder $f --case der_rich --reps 5 --no-sweep --kf $kf --tag _kf$kf \
      > results/log_${f}_der_rich_kf$kf.txt 2>&1
  done
done
echo SENS_DONE > results/sens_done.txt
