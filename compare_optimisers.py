#!/usr/bin/env python3
"""
Optimiser comparison on the robust group-coordination problem, Eq. (5):
WOA against PSO, GWO and DE with an identical budget (40 agents x 400
iterations), 30 independent runs per algorithm and problem, scored against
the exact LP reference. Problems (replication-0 training set of each feeder
and DER case): the single robust group (K = 1) and the two validity-selected
groups. Usage:  python compare_optimisers.py [--runs 30]
Writes results/optimisers.json and prints a summary table.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.stats import mannwhitneyu

import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
NA, NI = R.N_AGENTS, R.N_ITERS


def clip(X):
    return np.clip(X, R.TMS_LO, R.TMS_HI)


def pso(fit, dim, seed, w=0.72, c1=1.49, c2=1.49):
    """Inertia-weight PSO (Kennedy & Eberhart, 1995) with the widely used
    constants w = 0.72, c1 = c2 = 1.49; velocity clamped to 20 % of the range."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(R.TMS_LO, R.TMS_HI, (NA, dim)); V = np.zeros_like(X)
    vmax = 0.2 * (R.TMS_HI - R.TMS_LO)
    f = fit(X); Pb, fP = X.copy(), f.copy(); g = int(np.argmin(f)); G, fG = X[g].copy(), f[g]
    for _ in range(NI):
        r1, r2 = rng.random((NA, dim)), rng.random((NA, dim))
        V = np.clip(w * V + c1 * r1 * (Pb - X) + c2 * r2 * (G - X), -vmax, vmax)
        X = clip(X + V); f = fit(X)
        imp = f < fP; Pb[imp], fP[imp] = X[imp], f[imp]
        g = int(np.argmin(fP))
        if fP[g] < fG:
            G, fG = Pb[g].copy(), fP[g]
    return G, fG


def gwo(fit, dim, seed):
    """Grey wolf optimiser (Mirjalili et al., 2014)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(R.TMS_LO, R.TMS_HI, (NA, dim)); f = fit(X)
    order = np.argsort(f); L = X[order[:3]].copy(); fL = f[order[:3]].copy()
    for t in range(NI):
        a = 2.0 - 2.0 * t / NI
        Xn = np.zeros_like(X)
        for k in range(3):
            A = 2 * a * rng.random((NA, dim)) - a; C = 2 * rng.random((NA, dim))
            Xn += L[k] - A * np.abs(C * L[k] - X)
        X = clip(Xn / 3.0); f = fit(X)
        allX = np.vstack([L, X]); allf = np.concatenate([fL, f])
        order = np.argsort(allf); L, fL = allX[order[:3]].copy(), allf[order[:3]].copy()
    return L[0], fL[0]


def de(fit, dim, seed, F=0.5, CR=0.9):
    """Differential evolution DE/rand/1/bin (Storn & Price, 1997)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(R.TMS_LO, R.TMS_HI, (NA, dim)); f = fit(X)
    for _ in range(NI):
        idx = np.array([rng.choice(np.delete(np.arange(NA), i), 3, replace=False) for i in range(NA)])
        Vm = clip(X[idx[:, 0]] + F * (X[idx[:, 1]] - X[idx[:, 2]]))
        cross = rng.random((NA, dim)) < CR
        cross[np.arange(NA), rng.integers(0, dim, NA)] = True
        U = np.where(cross, Vm, X); fu = fit(U)
        better = fu <= f; X[better], f[better] = U[better], fu[better]
    i = int(np.argmin(f))
    return X[i], f[i]


def woa(fit, dim, seed):
    T, f, _ = R.woa(fit, dim, seed=seed)
    return T, f


ALGOS = {"WOA": woa, "PSO": pso, "GWO": gwo, "DE": de}


def problems(feeder, case):
    R.set_feeder(feeder); R.set_case(case)
    S = R.generate(480, 1000)                       # replication-0 training set
    L, P, B = R.currents(S)
    out = [("K=1", P, B, R.pickups(L))]
    X = R.cluster_features(S); Xs = R.stdz(X, X)
    _, best = R.validity(Xs)
    lab = R.cluster(Xs, best["method"], best["K"])
    for c in range(best["K"]):
        idx = lab == c
        out.append((f"group {c + 1}/{best['K']} (n={idx.sum()})", P[idx], B[idx], R.pickups(R.sel(L, idx))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    a = ap.parse_args()
    res = []
    for feeder in ("ieee33", "ieee69", "ieee123"):
        for case in ("moderate", "der_rich"):
            for name, P, B, pk in problems(feeder, case):
                f_lp = R.lp_reference(P, B, pk)
                fit = lambda X, P=P, B=B, pk=pk: R.fitness_pop(X, P, B, pk)
                row = dict(feeder=feeder, case=case, problem=name, n_states=int(P.shape[0]), lp=f_lp, algos={})
                vals = {}
                for an, fn in ALGOS.items():
                    fs, ok, ts = [], [], []
                    for r in range(a.runs):
                        t0 = time.perf_counter()
                        T, fv = fn(fit, R.NREL, 50000 + 97 * r)
                        ts.append(time.perf_counter() - t0)
                        fs.append(float(R.fitness_pop(T[None], P, B, pk)[0]))
                        tp = R.op_time(T[R.P_IDX][None], P, pk[R.P_IDX]); tb = R.op_time(T[R.B_IDX][None], B, pk[R.B_IDX])
                        with np.errstate(invalid="ignore"):
                            ok.append(bool(np.all(np.isfinite(tp)) and np.all(np.isfinite(tb)) and np.all(tb - tp >= R.CTI - 1e-9)))
                    fs = np.array(fs); vals[an] = fs
                    gap = 100 * (fs - f_lp) / f_lp if np.isfinite(f_lp) else np.full_like(fs, np.nan)
                    row["algos"][an] = dict(best=float(fs.min()), mean=float(fs.mean()), std=float(fs.std(ddof=1)),
                                            worst=float(fs.max()), gap_mean=float(np.nanmean(gap)),
                                            gap_best=float(np.nanmin(gap)), within_05=float(100 * np.nanmean(gap <= 0.5)),
                                            success=float(100 * np.mean(ok)), time_s=float(np.mean(ts)))
                for an in ("PSO", "GWO", "DE"):
                    p = mannwhitneyu(vals["WOA"], vals[an], alternative="two-sided").pvalue
                    row["algos"][an]["p_vs_woa"] = float(p)
                res.append(row)
                print(f"{feeder:8s} {case:9s} {name:22s} LP={f_lp:7.4f} | " + " | ".join(
                    f"{an}: mean {v['mean']:.4f}±{v['std']:.4f} gap {v['gap_mean']:+.2f}% ok {v['success']:.0f}% {v['time_s']:.2f}s"
                    + (f" p={v['p_vs_woa']:.1e}" if 'p_vs_woa' in v else "") for an, v in row["algos"].items()), flush=True)
    with open(os.path.join(HERE, "results", "optimisers.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()
