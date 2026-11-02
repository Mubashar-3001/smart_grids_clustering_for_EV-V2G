#!/usr/bin/env python3
"""
Clustering-method comparison: k-means, Ward, k-medoids and a Gaussian
mixture model (GMM), K = 2..8, scored with the Silhouette, Davies-Bouldin and
Calinski-Harabasz indices on the replication-0 training states of each DER
case. The standardised operating-state features do not depend on the feeder,
so the partitions are shared by the three feeders; pickups and grading are
not. For every feeder and method, two choices of K are carried through group
design (robust WOA) and scored on the 480 unseen states:
  * validity-only: the method's silhouette-best K;
  * certified: the method's candidates in silhouette order, first one whose
    every group passes the exact LP gradability certificate (K = 1 if none).
Usage: python compare_clustering.py      -> results/clustering.json
"""
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

import run_experiment as R

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("kmeans", "ward", "kmedoids", "gmm")


def kmedoids(X, K, seed=0, n_init=10, iters=100):
    """Alternating k-medoids (Park & Jun, 2009), Euclidean distance.
    Returns (labels, medoid indices)."""
    rng = np.random.default_rng(seed)
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(2))
    best, best_cost = None, np.inf
    for _ in range(n_init):
        med = rng.choice(len(X), K, replace=False)
        for _ in range(iters):
            lab = np.argmin(D[:, med], 1)
            new = np.array([np.where(lab == c)[0][np.argmin(D[np.ix_(lab == c, lab == c)].sum(1))]
                            if np.any(lab == c) else med[c] for c in range(K)])
            if np.all(new == med):
                break
            med = new
        cost = D[np.arange(len(X)), med[np.argmin(D[:, med], 1)]].sum()
        if cost < best_cost:
            best_cost, best = cost, (np.argmin(D[:, med], 1), med.copy())
    return best


def labels(Xs, method, K):
    """Training labels (contiguous 0..K'-1) and the method's own rule for
    assigning new states: nearest centroid (k-means, Ward), nearest medoid
    (k-medoids) or maximum posterior (GMM)."""
    if method == "kmeans":
        km = KMeans(K, n_init=10, random_state=0).fit(Xs)
        lab, assign = km.labels_, km.predict
    elif method == "ward":
        lab = R.cluster(Xs, "ward", K)
        cen = np.array([Xs[lab == j].mean(0) for j in range(K)])
        assign = lambda Z, cen=cen: R.nearest(cen, Z)
    elif method == "kmedoids":
        lab, med = kmedoids(Xs, K)
        assign = lambda Z, M=Xs[med]: R.nearest(M, Z)
    else:
        gm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=0).fit(Xs)
        lab, assign = gm.predict(Xs), gm.predict
    u, lab = np.unique(lab, return_inverse=True)
    remap = {int(v): i for i, v in enumerate(u)}
    return lab, (lambda Z, f=assign: np.array([remap.get(int(v), 0) for v in f(Z)]))


def design_and_score(lab, assign, Xs_te, Ltr, Ptr, Btr, Lte, Pte, Bte, T1, pk1):
    K = int(lab.max()) + 1
    if K == 1:
        T, PK = T1[None], pk1[None]
        a = np.zeros(len(Xs_te), int)
    else:
        T, PK, _ = R.group_settings(lab, K, Ltr, Ptr, Btr, 777)
        a = assign(Xs_te)
    e = R.evaluate(T[a], PK[a], Lte, Pte, Bte)
    return dict(K=K, sizes=np.bincount(lab).tolist(), affected=e["affected"], min_margin=e["min_margin"],
                pk_trunk=e["pk_trunk"], t_prim=e["t_prim"])


def main():
    out = {}
    for case in ("moderate", "der_rich"):
        R.set_feeder("ieee33"); R.set_case(case)
        Str, Ste = R.generate(480, 1000), R.generate(480, 5000)
        Xtr, Xte = R.cluster_features(Str), R.cluster_features(Ste)
        Xs, Xs_te = R.stdz(Xtr, Xtr), R.stdz(Xtr, Xte)
        table, labs = [], {}
        for m in METHODS:
            for K in range(2, 9):
                lab, assign = labels(Xs, m, K)
                if lab.max() < 1:
                    continue
                labs[(m, K)] = (lab, assign)
                table.append(dict(method=m, K=K, K_eff=int(lab.max()) + 1, sil=float(silhouette_score(Xs, lab)),
                                  db=float(davies_bouldin_score(Xs, lab)), ch=float(calinski_harabasz_score(Xs, lab))))
        res = dict(validity=table, outcome={})
        for feeder in ("ieee33", "ieee69", "ieee123"):
            R.set_feeder(feeder); R.set_case(case)
            Ltr, Ptr, Btr = R.currents(Str); Lte, Pte, Bte = R.currents(Ste)
            pk1 = R.pickups(Ltr); T1, _ = R.optimise(Ptr, Btr, pk1, 778)
            res["outcome"][feeder] = {}
            for m in METHODS:
                rows = sorted([r for r in table if r["method"] == m], key=lambda r: (-r["sil"], r["db"]))
                v = rows[0]
                val = design_and_score(*labs[(m, v["K"])], Xs_te, Ltr, Ptr, Btr, Lte, Pte, Bte, T1, pk1)
                cert, n_checked = None, 0
                for r in rows:
                    n_checked += 1
                    lab = labs[(m, r["K"])][0]
                    if all(R.gradable(Ptr[lab == g], Btr[lab == g], R.pickups(R.sel(Ltr, lab == g)))
                           for g in range(int(lab.max()) + 1)):
                        cert = r
                        break
                lab_c, as_c = labs[(m, cert["K"])] if cert else (np.zeros(len(Xs), int), None)
                cer = design_and_score(lab_c, as_c, Xs_te, Ltr, Ptr, Btr, Lte, Pte, Bte, T1, pk1)
                res["outcome"][feeder][m] = dict(validity_only=dict(val, sil=v["sil"], db=v["db"], ch=v["ch"]),
                                                 certified=dict(cer, n_checked=n_checked,
                                                                sil=cert["sil"] if cert else float("nan")))
                print(f"{case:9s} {feeder:8s} {m:9s} | validity-only K={val['K']} (sil {v['sil']:.3f}, DB {v['db']:.3f}, "
                      f"CH {v['ch']:.0f}) sizes={val['sizes']} -> unseen {val['affected']:.1f}% margin {val['min_margin']:.3f} s "
                      f"pk {val['pk_trunk']:.1f} A | certified K={cer['K']} after {n_checked} -> unseen {cer['affected']:.1f}% "
                      f"margin {cer['min_margin']:.3f} s pk {cer['pk_trunk']:.1f} A", flush=True)
        out[case] = res
    with open(os.path.join(HERE, "results", "clustering.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
