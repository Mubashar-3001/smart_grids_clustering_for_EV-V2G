#!/usr/bin/env python3
"""Prints every number quoted in the paper, traced to the result files.
Run after run_all.sh, run_sensitivity.sh, compare_optimisers.py and
compare_clustering.py:
    python paper_numbers.py > results/paper_numbers.txt"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FEEDERS = ("ieee33", "ieee69", "ieee123")


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(d, k, p=1):
    return f"{d[k]['mean']:.{p}f} +/- {d[k]['std']:.{p}f}"


# ------------------------------------------------------------------ main study
tot_unseen = 0
for f in FEEDERS:
    for case in ("moderate", "der_rich"):
        s = load(f"summary_{f}_{case}.json")
        if s is None:
            continue
        c = s["config"]; tot_unseen += c["reps"] * c["test"]
        print(f"\n=========== {f} | {case} ({c['reps']} replications x {c['train']} training / {c['test']} unseen) ===========")
        print("certified K:", s["K_selected"], s["methods"], f"| mean Sil {s['sil_mean']:.3f}, DB {s['db_mean']:.3f}")
        print("validity-best K:", s["K_validity"], s["methods_validity"], "| ungradable groups in it:",
              s["ungradable_validity"], "| candidates checked:", s["n_checked"], f"| mean Sil {s['sil_validity_mean']:.3f}")
        print(f"DER-induced change of pair fault currents (max over pairs, max/min - 1): {s['fc_spread_pct_max']:.1f} %")
        for n, v in s["schemes"].items():
            extra = ""
            if "acc" in v:
                extra += f"  kNN acc {100 * v['acc']['mean']:.1f} +/- {100 * v['acc']['std']:.1f} %"
            if "fallback_pct" in v:
                extra += f"  fallback {v['fallback_pct']['mean']:.2f} %"
            print(f"  {n:17s} groups {v['groups']['mean']:4.1f} | events {fmt(v, 'total')} "
                  f"(CTI {v['cti']['mean']:.1f}, blind {v['blind']['mean']:.1f}, enc {v['enc']['mean']:.1f}) | "
                  f"affected {fmt(v, 'affected', 2)} % | worst margin {v['min_margin']['min']:.3f} s | "
                  f"t_prim {fmt(v, 't_prim', 4)} s | trunk pickup {fmt(v, 'pk_trunk')} A{extra}")
        k1 = s["schemes"]["robust_K1"]["pk_trunk"]["mean"]
        pr = s["schemes"]["proposed_knn_env"]["pk_trunk"]["mean"]
        print(f"  trunk pickup, proposed (kNN+env) vs K=1: {100 * (1 - pr / k1):.1f} % lower over all unseen states;"
              f" largest regime reduction {s['pk_red_regime_max']:.1f} %; PV-export regime {s['pk_red_pv_pct']:.1f} %"
              f" ({s['pv_cluster_share_test']:.1f} % of unseen states)")
        print("  in-sample events (proposed):", s["insample_total"])
        g = np.array([x for rr in s["reps"] for x in rr["meta"]["woa_lp_gap_pct"]], float)
        print(f"  WOA vs LP optimum: {np.isfinite(g).sum()} solves, gap min {np.nanmin(g):.3f} %, median {np.nanmedian(g):.3f} %, "
              f"max {np.nanmax(g):.3f} %, share <= 0.5 %: {100 * np.nanmean(g <= 0.5):.0f} %")
        print(f"  LP-optimal TMS for the same groups: affected {s['lp_check']['affected']:.2f} %, worst margin "
              f"{s['lp_check_min_margin']:.3f} s, t_prim {s['lp_check']['t_prim']:.4f} s (WOA {s['lp_check']['t_prim_woa']:.4f} s)")
        print(f"  WOA convergence (iterations to within 0.1 % of final, max): {s['conv_iters_max']}")
        print(f"  kNN inference {s['knn_ms_per_sample'] * 1e3:.1f} us/state; runtime per replication {s['t_rep_s']:.1f} s")
        print("  sweep (mean over replications):")
        for k, v in s["sweep"].items():
            print(f"     {k:11s} ungradable {v['ungradable']['mean']:.1f}  in-sample {v['in_aff']['mean']:5.1f} %  "
                  f"unseen {v['out_aff']['mean']:5.1f} +/- {v['out_aff']['std']:4.1f} %  trunk pk {v['pk_trunk']['mean']:.1f} A")
print(f"\nTOTAL unseen states in the main study: {tot_unseen}")

# ------------------------------------------------------------------ sensitivity
print("\n=========== sensitivity (DER-rich, 5 replications) ===========")
for f in FEEDERS:
    for tag, lab in (("_zs0.10", "Xs=0.10"), ("_zs0.15", "Xs=0.15"), ("", "Xs=0.20, kf=2.0 (base, 10 reps)"), ("_zs0.40", "Xs=0.40"),
                     ("_kf1.2", "kf=1.2"), ("_kf1.5", "kf=1.5")):
        s = load(f"summary_{f}_der_rich{tag}.json")
        if s is None:
            continue
        sc = s["schemes"]
        print(f"  {f:8s} {lab:32s} K={s['K_selected']} | conventional {sc['conventional']['affected']['mean']:5.1f} % "
              f"(worst {sc['conventional']['min_margin']['min']:.3f} s) | robust K=1 {sc['robust_K1']['affected']['mean']:.1f} % | "
              f"proposed {sc['proposed_knn_env']['affected']['mean']:.1f} % (worst {sc['proposed_knn_env']['min_margin']['min']:.3f} s) | "
              f"scalar-11 {sc['scalar_11']['affected']['mean']:.1f} % | trunk pk K=1 {sc['robust_K1']['pk_trunk']['mean']:.0f} A, "
              f"proposed {sc['proposed_knn_env']['pk_trunk']['mean']:.0f} A | fc change {s['fc_spread_pct_max']:.1f} %")

# ------------------------------------------------------------------ optimisers
res = load("optimisers.json")
if res:
    print("\n=========== optimisers (30 runs, 40 x 400 budget) ===========")
    feas = [r for r in res if np.isfinite(r["lp"])]
    for r in res:
        if not np.isfinite(r["lp"]):
            print("  LP-ungradable problem:", r["feeder"], r["case"], r["problem"],
                  {a: (round(v["mean"], 2), v["success"]) for a, v in r["algos"].items()})
    print(f"  gradable problems: {len(feas)}")
    for a in ("WOA", "PSO", "GWO", "DE"):
        gm = np.array([r["algos"][a]["gap_mean"] for r in feas])
        gw = np.array([100 * (r["algos"][a]["worst"] - r["lp"]) / r["lp"] for r in feas])
        w5 = np.array([r["algos"][a]["within_05"] for r in feas])
        sc = np.array([r["algos"][a]["success"] for r in feas])
        t = np.array([r["algos"][a]["time_s"] for r in feas])
        line = (f"  {a}: mean gap {gm.mean():.2f} % (median over problems {np.median(gm):.2f} %, largest {gm.max():.2f} %), "
                f"worst single run {gw.max():.1f} %, runs within 0.5 %: {w5.mean():.1f} %, coordinated runs {sc.mean():.1f} % "
                f"(lowest {sc.min():.0f} %), time {t.mean():.2f} s/run")
        if a != "WOA":
            b = sum(r["algos"][a]["mean"] < r["algos"]["WOA"]["mean"] and r["algos"][a]["p_vs_woa"] < 0.05 for r in feas)
            w = sum(r["algos"][a]["mean"] > r["algos"]["WOA"]["mean"] and r["algos"][a]["p_vs_woa"] < 0.05 for r in feas)
            line += f" | Mann-Whitney vs WOA (p < 0.05): better {b}, worse {w}, no difference {len(feas) - b - w}"
        print(line)

# ------------------------------------------------------------------ clustering
clu = load("clustering.json")
if clu:
    print("\n=========== clustering methods (replication 0) ===========")
    for case, res in clu.items():
        for m in ("kmeans", "ward", "kmedoids", "gmm"):
            rows = [r for r in res["validity"] if r["method"] == m]
            b = sorted(rows, key=lambda r: (-r["sil"], r["db"]))[0]
            print(f"  {case:9s} {m:9s} best Sil {b['sil']:.3f} (K={b['K']}), DB {b['db']:.3f}, CH {b['ch']:.0f}")
        for f, o in res["outcome"].items():
            for m, v in o.items():
                print(f"     {case:9s} {f:8s} {m:9s} validity-only K={v['validity_only']['K']} sizes={v['validity_only']['sizes']} "
                      f"-> {v['validity_only']['affected']:.1f} % (worst {v['validity_only']['min_margin']:.3f} s, pk {v['validity_only']['pk_trunk']:.1f} A)"
                      f" | certified K={v['certified']['K']} after {v['certified']['n_checked']} -> {v['certified']['affected']:.1f} % "
                      f"(pk {v['certified']['pk_trunk']:.1f} A)")
