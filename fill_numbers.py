#!/usr/bin/env python3
"""Fills the {{NAME}} placeholders of the journal text (sec_results.tex,
sec_front.tex) with numbers computed from the result files, so that every
number in the prose is traced to results/*.json. Prints the mapping.
Usage: python fill_numbers.py   -> ../journal_src/sec_results_filled.tex, sec_front_filled.tex"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, J = os.path.join(HERE, "results"), os.path.join(HERE, "..", "journal_src")
F = ("ieee33", "ieee69", "ieee123")
C = ("moderate", "der_rich")
S = {(f, c): json.load(open(os.path.join(RES, f"summary_{f}_{c}.json"))) for f in F for c in C}
sc = lambda f, c, k, m="affected", a="mean": S[(f, c)]["schemes"][k][m][a]


def rng(vals, p=1, sep="--"):
    lo, hi = min(vals), max(vals)
    return f"{lo:.{p}f}{sep}{hi:.{p}f}" if round(lo, p) != round(hi, p) else f"{lo:.{p}f}"


v = {}
v["SIL"] = f"{np.mean([S[(f, c)]['sil_validity_mean'] for f in F for c in C]):.2f}"
v["DB"] = f"{np.mean([S[('ieee33', c)]['db_mean'] for c in C]):.2f}"
v["PVSHARE"] = f"{np.mean([S[('ieee33', c)]['pv_cluster_share_test'] for c in C]):.0f}"
held = [(f, c) for f in F for c in C if sc(f, c, "conventional") == 0.0]
v["CONVEROSION33"] = f"{1e3 * max(0.310 - sc(f, c, 'conventional', 'min_margin', 'min') for f, c in held):.0f}"
v["CONV69"] = f"{sc('ieee69', 'der_rich', 'conventional'):.1f}"
v["CONV123"] = f"{sc('ieee123', 'der_rich', 'conventional'):.1f}"
v["CONVM69"] = f"{sc('ieee69', 'der_rich', 'conventional', 'min_margin', 'min'):.3f}"
v["CONVM123"] = f"{sc('ieee123', 'der_rich', 'conventional', 'min_margin', 'min'):.3f}"
v["CONVMAXR"] = f"{max(sc(f, 'der_rich', 'conventional') for f in F):.0f}"
two = [(f, c) for f in F for c in C if np.median(S[(f, c)]["K_selected"]) >= 2]
red_pv = [S[k]["pk_red_pv_pct"] for k in two]
red_all = [100 * (1 - sc(*k, "proposed_knn_env", "pk_trunk") / sc(*k, "robust_K1", "pk_trunk")) for k in two]
v["PKRED"] = rng(red_pv, 1, r"--") + r"\%"
v["PKALL"] = rng(red_all, 1, r"--") + r"\%"
v["PKMAX"] = f"{max(red_pv):.1f}"
def regime(f, c, which):
    reps = [rr for rr in S[(f, c)]["reps"] if rr["meta"]["K"] == 2]
    p = [rr["meta"]["pv_cluster"] for rr in reps]
    return np.mean([rr["meta"]["pk_red_regimes"][pc if which == "pv" else 1 - pc] for rr, pc in zip(reps, p)])
v["PV33M"] = f"{regime('ieee33', 'moderate', 'pv'):.1f}"
v["PV69M"] = f"{regime('ieee69', 'moderate', 'pv'):.1f}"
v["OTH123"] = f"{regime('ieee123', 'der_rich', 'other'):.1f}"
all_red = {k: 100 * (1 - sc(*k, "proposed_knn_env", "pk_trunk") / sc(*k, "robust_K1", "pk_trunk")) for k in S}
v["ALL12"] = rng([all_red[(f, c)] for f in ("ieee33", "ieee69") for c in C], 1)
v["ALL123"] = f"{all_red[('ieee123', 'der_rich')]:.1f}"
tps = [sc(f, c, k, "t_prim") for f in F for c in C for k in ("robust_K1", "proposed_knn_env")]
v["TPMIN"], v["TPMAX"] = f"{min(tps):.2f}", f"{max(tps):.2f}"
v["UNG123"] = f"{sum(x > 0 for x in S[('ieee123', 'moderate')]['ungradable_validity'])}"
v["VAL123"] = f"{sc('ieee123', 'moderate', 'validity_only'):.1f}"
v["VALM123"] = f"{sc('ieee123', 'moderate', 'validity_only', 'min_margin', 'min'):.3f}"
acc = [100 * sc(*k, "proposed_knn", "acc") for k in two]
v["KNNMIN"], v["KNNMAX"] = f"{min(acc):.1f}", f"{max(acc):.1f}"
v["FALLBACK"] = f"{max(sc(f, c, 'proposed_knn_env', 'fallback_pct') for f in F for c in C):.2f}"
sw = lambda f, c, k: S[(f, c)]["sweep"][k]["out_aff"]["mean"]
v["SWEEP4MIN"] = f"{min(sw(f, c, 'cluster_4') for f in F for c in C):.0f}"
v["SWEEP4MAX"] = f"{max(sw(f, c, 'cluster_4') for f in F for c in C):.0f}"
v["SWEEP8MAX"] = f"{max(sw(f, c, k) for f in F for c in C for k in ('cluster_4', 'cluster_5', 'cluster_6', 'cluster_8')):.0f}"
v["SWEEPMAX"] = v["SWEEP8MAX"]
sc11 = [sc(f, c, "scalar_11") for f in F for c in C]
v["SC11MIN"], v["SC11MAX"] = f"{min(sc11):.0f}", f"{max(sc11):.0f}"
v["FC33"], v["FC69"], v["FC123"] = (f"{S[(f, 'der_rich')]['fc_spread_pct_max']:.1f}" for f in F)
v["XSLIST"] = "0.10, 0.15 and 0.40"

for name in ("sec_results", "sec_front"):
    txt = open(os.path.join(J, f"{name}.tex")).read()
    for k, val in v.items():
        txt = txt.replace("{{" + k + "}}", val)
    left = sorted(set(re.findall(r"\{\{([A-Z0-9]+)\}\}", txt)))
    open(os.path.join(J, f"{name}_filled.tex"), "w").write(txt)
    print(f"{name}: unfilled placeholders -> {left}")
for k, val in v.items():
    print(f"  {k:14s} = {val}")
