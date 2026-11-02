#!/bin/bash
# Extra grid-strength point (Xs = 0.15 pu) between the infeasible 0.10 pu and the base 0.20 pu.
cd "$(dirname "$0")"
for f in ieee33 ieee69 ieee123; do
  python3 run_experiment.py --feeder $f --case der_rich --reps 5 --no-sweep --zs 0.15 --tag _zs0.15 \
    > results/log_${f}_der_rich_zs0.15.txt 2>&1
done
echo SENS2_DONE > results/sens2_done.txt
