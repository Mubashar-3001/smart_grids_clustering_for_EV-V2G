#!/usr/bin/env python3
"""LaTeX tables for the journal paper, generated from the result files so that
no number is typed by hand. Run after run_all.sh, compare_optimisers.py and
compare_clustering.py:
    python make_tables.py      -> ../journal_src/tab_results.tex, tab_optimizers.tex, tab_clustering.tex
Grey shading marks the best entry (user style: shading, not bold)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "..", "journal_src")
FEEDERS = (("ieee33", "33-bus"), ("ieee69", "69-bus"), ("ieee123", "123-bus"))
SCHEMES = (("conventional", "Conventional (peak load, no DER)"),
           ("robust_K1", "Single robust group ($K=1$)"),
           ("validity_only", "Validity-only groups"),
           ("proposed_knn_env", "Proposed, $k$-NN + envelope"),
           ("scalar_K", "Scalar index, same $K$"),
           ("scalar_11", "Scalar index, 11 levels"))
SH = r"\cellcolor{gray!20}"


def load(n):
    return json.load(open(os.path.join(RES, n)))


def groups_label(vals):
    """Most frequent group count over the replications (an asterisk marks
    configurations in which other counts also occurred)."""
    v = [int(round(x)) for x in vals]
    mode = max(set(v), key=v.count)
    return str(mode) + ("$^{*}$" if len(set(v)) > 1 else "")


# ------------------------------------------------------------ main results
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Unseen-state results, mean of ten replications $\times$ 480 states per feeder and DER case. "
     r"$n_g$: number of setting groups (33/69/123-bus where they differ; $^{*}$two groups were accepted in one replication of ten); "
     r"the envelope fallback of the proposed design adds the single robust group. "
     r"Misc.: share of states with at least one miscoordination; Margin: worst pair margin over all 4\,800 states; "
     r"$I_p$: mean trunk pickup. Shading marks the lowest trunk pickup among designs with no miscoordination.}",
     r"\label{tab:results}", r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}",
     r"\begin{tabular}{@{}l c rcr rcr rcr@{}}", r"\toprule",
     r" & & \multicolumn{3}{c}{IEEE 33-bus} & \multicolumn{3}{c}{IEEE 69-bus} & \multicolumn{3}{c}{IEEE 123-bus} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
     r"Design & $n_g$ & Misc. & Margin & $I_p$ & Misc. & Margin & $I_p$ & Misc. & Margin & $I_p$ \\",
     r" & & (\%) & (s) & (A) & (\%) & (s) & (A) & (\%) & (s) & (A) \\", r"\midrule"]
for case, cname in (("moderate", "Moderate DER"), ("der_rich", "DER-rich")):
    S = {f: load(f"summary_{f}_{case}.json") for f, _ in FEEDERS}
    L.append(r"\multicolumn{11}{@{}l}{\emph{" + cname + r"}} \\")
    best = {}
    for f, _ in FEEDERS:
        sec = [(S[f]["schemes"][k]["pk_trunk"]["mean"], k) for k, _ in SCHEMES
               if S[f]["schemes"][k]["affected"]["mean"] == 0.0]
        best[f] = min(sec)[1] if sec else None
    for key, name in SCHEMES:
        ng = []
        for f, _ in FEEDERS:
            g = [rr["schemes"][key]["groups"] - (key == "proposed_knn_env" and rr["schemes"][key]["groups"] > 1)
                 for rr in S[f]["reps"]]
            ng.append(groups_label(g))
        ngs = ng[0] if len(set(ng)) == 1 else "/".join(ng)
        row = [name, ngs]
        for f, _ in FEEDERS:
            v = S[f]["schemes"][key]
            a, m, p = v["affected"]["mean"], v["min_margin"]["min"], v["pk_trunk"]["mean"]
            row += [f"{a:.1f}", f"{m:.3f}", (SH if best[f] == key else "") + f"{p:.1f}"]
        L.append(" & ".join(row) + r" \\")
    if case == "moderate":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_results.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------ optimisers
res = load("optimisers.json")
feas = [r for r in res if np.isfinite(r["lp"])]
rows = {}
for a in ("WOA", "PSO", "GWO", "DE"):
    gm = np.array([r["algos"][a]["gap_mean"] for r in feas])
    gw = np.array([100 * (r["algos"][a]["worst"] - r["lp"]) / r["lp"] for r in feas])
    rows[a] = dict(mean=gm.mean(), med=np.median(gm), worst=gw.max(),
                   w05=np.mean([r["algos"][a]["within_05"] for r in feas]),
                   ok=np.mean([r["algos"][a]["success"] for r in feas]),
                   t=np.mean([r["algos"][a]["time_s"] for r in feas]))
    if a != "WOA":
        rows[a]["better"] = sum(r["algos"][a]["mean"] < r["algos"]["WOA"]["mean"] and r["algos"][a]["p_vs_woa"] < 0.05 for r in feas)
        rows[a]["worse"] = sum(r["algos"][a]["mean"] > r["algos"]["WOA"]["mean"] and r["algos"][a]["p_vs_woa"] < 0.05 for r in feas)
for a in rows:                                   # gaps below zero are rounding/penalty noise
    rows[a]["med"] = max(rows[a]["med"], 0.0); rows[a]["mean"] = max(rows[a]["mean"], 0.0)
fmt = {"mean": 2, "med": 2, "worst": 1, "w05": 1, "ok": 1, "t": 2}
best_of = lambda k, f: f(round(rows[a][k], fmt[k]) for a in rows)
bestcol = {k: {a for a in rows if round(rows[a][k], fmt[k]) == best_of(k, min if k in ("mean", "med", "worst", "t") else max)}
           for k in fmt}
nprob = len(feas)
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Metaheuristics against the LP optimum on the " + str(nprob) + r" gradable group-design problems "
     r"(three feeders, two DER cases, the single robust group and the validity-selected groups of replication~0); "
     r"30 runs each, 40 agents $\times$ 400 iterations. Gaps are in the robust objective of Eq.~\eqref{eq:robust}; "
     r"the last column counts problems in which the algorithm was significantly better or worse than WOA "
     r"(two-sided Mann--Whitney test, $p<0.05$). Shading marks the best value in each column.}",
     r"\label{tab:optimizers}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{@{}lrrrrrrc@{}}", r"\toprule",
     r"Algorithm & Mean gap & Median gap & Worst run & Runs within & Coordinated & Time per & vs WOA \\",
     r" & (\%) & (\%) & (\%) & 0.5\% (\%) & runs (\%) & run (s) & better/worse \\", r"\midrule"]
names = {"WOA": r"WOA~\cite{woa2016}", "PSO": r"PSO~\cite{pso1995}", "GWO": r"GWO~\cite{gwo2014}", "DE": r"DE~\cite{de1997}"}
for a in ("WOA", "PSO", "GWO", "DE"):
    r = rows[a]
    c = lambda k, s: (SH if a in bestcol[k] else "") + s
    vs = "--" if a == "WOA" else f"{r['better']}/{r['worse']}"
    L.append(" & ".join([names[a], c("mean", f"{r['mean']:.2f}"), c("med", f"{r['med']:.2f}"),
                         c("worst", f"{r['worst']:.1f}"), c("w05", f"{r['w05']:.1f}"),
                         c("ok", f"{r['ok']:.1f}"), c("t", f"{r['t']:.2f}"), vs]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_optimizers.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------ clustering
clu = load("clustering.json")
mname = {"kmeans": r"$k$-means~\cite{lloyd1982}", "ward": r"Ward~\cite{ward1963}",
         "kmedoids": r"$k$-medoids~\cite{parkjun2009}", "gmm": r"GMM (EM)~\cite{dempster1977}"}
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Clustering algorithms on the replication-0 states. Validity indices are for each method's "
     r"silhouette-best partition (DER-rich case; partitions do not depend on the feeder). Unseen-state results use "
     r"each method's own assignment rule; 123-bus moderate: miscoordination without the certificate and the group count "
     r"the certificate accepts; 123-bus DER-rich: mean trunk pickup of the certified groups. Shading marks the best value.}",
     r"\label{tab:clustering}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{Validity (DER-rich)} & \multicolumn{2}{c}{123-bus, moderate} & 123-bus, DER-rich \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(l){8-8}",
     r"Method & $K$ & Silhouette & Davies--Bouldin & Calinski--Harabasz & Misc.\ without cert.\ (\%) & Certified $K$ & $I_p$ (A) \\",
     r"\midrule"]
vals = {}
for m in ("kmeans", "ward", "kmedoids", "gmm"):
    rows_m = [r for r in clu["der_rich"]["validity"] if r["method"] == m]
    b = sorted(rows_m, key=lambda r: (-r["sil"], r["db"]))[0]
    o_mod = clu["moderate"]["outcome"]["ieee123"][m]
    o_dr = clu["der_rich"]["outcome"]["ieee123"][m]
    vals[m] = dict(K=b["K"], sil=b["sil"], db=b["db"], ch=b["ch"], mis=o_mod["validity_only"]["affected"],
                   kc=o_mod["certified"]["K"], pk=o_dr["certified"]["pk_trunk"])
bm = {"sil": max(vals, key=lambda m: vals[m]["sil"]), "db": min(vals, key=lambda m: vals[m]["db"]),
      "ch": max(vals, key=lambda m: vals[m]["ch"]), "mis": min(vals, key=lambda m: vals[m]["mis"]),
      "pk": min(vals, key=lambda m: vals[m]["pk"])}
for m, v in vals.items():
    c = lambda k, s: (SH if bm.get(k) == m else "") + s
    L.append(" & ".join([mname[m], str(v["K"]), c("sil", f"{v['sil']:.3f}"), c("db", f"{v['db']:.3f}"),
                         c("ch", f"{v['ch']:.0f}"), c("mis", f"{v['mis']:.1f}"), str(v["kc"]),
                         c("pk", f"{v['pk']:.1f}")]) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_clustering.tex"), "w").write("\n".join(L) + "\n")
print("tables written to", OUT)
