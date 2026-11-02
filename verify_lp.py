#!/usr/bin/env python3
"""Cross-check of the exact LP certificate (Eq. (6)) against the bottom-up
least-TMS recursion on every group of every candidate partition (k-means and
Ward, K = 1..8) of the replication-0 training states, for all feeders and DER
cases. Usage: python verify_lp.py"""
import numpy as np

import run_experiment as R

agree = disagree = 0
worst = 0.0
for f in ("ieee33", "ieee69", "ieee123"):
    for case in ("moderate", "der_rich"):
        R.set_feeder(f); R.set_case(case)
        S = R.generate(480, 1000); L, P, B = R.currents(S)
        Xs = R.stdz(R.cluster_features(S), R.cluster_features(S))
        groups = [np.ones(len(P), bool)]
        for K in range(2, 9):
            for m in ("kmeans", "ward"):
                lab = R.cluster(Xs, m, K)
                groups += [lab == c for c in range(K)]
        nung = 0
        for idx in groups:
            pk = R.pickups(R.sel(L, idx))
            f_lp = R.lp_reference(P[idx], B[idx], pk)
            T = R.min_tms(P[idx], B[idx], pk)
            f_rec = np.inf if T is None else float(R.fitness_pop(T[None], P[idx], B[idx], pk)[0])
            nung += not np.isfinite(f_lp)
            if np.isfinite(f_lp) == np.isfinite(f_rec):
                agree += 1
                if np.isfinite(f_lp):
                    worst = max(worst, abs(f_rec - f_lp) / f_lp)
            else:
                disagree += 1
        print(f"{f:8s} {case:9s}: {len(groups)} groups, {nung} ungradable")
print(f"LP and recursion agree on {agree} groups, disagree on {disagree}; "
      f"largest relative difference of the optimum {worst:.1e}")
