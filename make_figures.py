#!/usr/bin/env python3
"""Journal figures from results/summary_<feeder>_<case>[tag].json,
results/rep0_<feeder>_<case>.npz and the real-data CSVs. Run after
run_all.sh and run_sensitivity.sh:
    python make_figures.py
Writes ../figures/fig_data_regimes.pdf, fig_group_sweep.pdf, fig_margins.pdf
and fig_sensitivity.pdf (TrueType fonts embedded)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, DATA, OUT = (os.path.join(HERE, "results"), os.path.join(HERE, "..", "data"),
                  os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cc2"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3de", "#efeeea"
FEEDERS = (("ieee33", "IEEE 33-bus"), ("ieee69", "IEEE 69-bus"), ("ieee123", "IEEE 123-bus"))
CASES = (("moderate", "moderate DER"), ("der_rich", "DER-rich"))
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.5,
    "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})


def tidy(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def summ(feeder, case, tag=""):
    p = os.path.join(RES, f"summary_{feeder}_{case}{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def rep0(feeder, case):
    return np.load(os.path.join(RES, f"rep0_{feeder}_{case}.npz"), allow_pickle=True)


# ---------------- measured inputs and the regimes they produce ----------------
pv = np.loadtxt(os.path.join(DATA, "ausgrid_c12_pv_pu.csv"), delimiter=",", skiprows=1)[:, 4:]
ld = np.loadtxt(os.path.join(DATA, "ausgrid_c12_load_shape.csv"), delimiter=",", skiprows=1)
h = np.arange(48) * 0.5
d = rep0("ieee33", "der_rich")                     # partitions are identical on all feeders
lab, hour, pvt = d["lab_validity"], d["hour_tr"], d["pv_tr"]
K = int(lab.max()) + 1
pvc = int(np.argmax([pvt[lab == c].mean() for c in range(K)]))

fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.25), gridspec_kw=dict(wspace=0.25))
ax = axs[0]; tidy(ax)
ax.fill_between(h, np.percentile(pv, 10, 0), np.percentile(pv, 90, 0), color=BLUE, alpha=0.16, lw=0)
ax.plot(h, np.median(pv, 0), color=BLUE, label="PV output: median, 10–90% band (366 days)")
ax.fill_between(h, ld.min(0), ld.max(0), color=ORANGE, alpha=0.14, lw=0)
ax.plot(h, ld.mean(0), color=ORANGE, ls="--", label="Load shape: mean, monthly range")
ax.set_ylabel("Per unit"); ax.set_xlabel("Hour of day")
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_ylim(0, 1.12)
ax.legend(loc="upper left", frameon=False, handlelength=1.8, borderaxespad=0.1)
ax.set_title("(a) Measured inputs, Ausgrid customer 12, 2011–12", loc="left", pad=3)
ax = axs[1]; tidy(ax)
edges = np.arange(0, 25)
share = [np.mean(lab[(hour >= a) & (hour < a + 1)] == pvc) if np.any((hour >= a) & (hour < a + 1)) else 0.0
         for a in edges[:-1]]
ax.bar(edges[:-1] + 0.5, np.array(share) * 100, width=0.82, color=BLUE, edgecolor="white", linewidth=0.4)
ax.set_ylabel("States in PV-export cluster (%)"); ax.set_xlabel("Hour of day")
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_ylim(0, 100)
ax.set_title(f"(b) PV-export cluster by hour (DER-rich, K = {K}, training states)", loc="left", pad=3)
fig.savefig(os.path.join(OUT, "fig_data_regimes.pdf"), bbox_inches="tight")
plt.close(fig)


# ---------------- how many setting groups? (three feeders, two cases) ---------
def sweep_series(s, kind):
    sw = s["sweep"]
    ns = sorted(int(k.split("_")[1]) for k in sw if k.startswith(kind))
    x = [1] + ns
    out = [s["schemes"]["robust_K1"]["affected"]["mean"]] + [sw[f"{kind}_{n}"]["out_aff"]["mean"] for n in ns]
    ins = [np.nan] + [sw[f"{kind}_{n}"]["in_aff"]["mean"] for n in ns]
    ung = [0.0] + [sw[f"{kind}_{n}"]["ungradable"]["mean"] for n in ns]
    return np.array(x), np.array(out), np.array(ins), np.array(ung)


fig, axs = plt.subplots(3, 3, figsize=(6.9, 6.0), sharex=True, gridspec_kw=dict(wspace=0.22, hspace=0.30))
for j, (fd, fname) in enumerate(FEEDERS):
    for i, (case, cname) in enumerate(CASES):
        ax = axs[i, j]; tidy(ax)
        s = summ(fd, case)
        kmode = int(np.round(np.median(s["K_selected"])))
        ax.axvspan(kmode - 0.35, kmode + 0.35, color=BAND, lw=0, zorder=0)
        for kind, col, mk, name in (("cluster", BLUE, "o", "Clustered"), ("scalar", ORANGE, "s", "Scalar index")):
            x, out, ins, _ = sweep_series(s, kind)
            ax.plot(x, out, color=col, marker=mk, ms=3.4, label=f"{name}, unseen states")
            ax.plot(x, ins, color=col, ls=":", lw=1.1, marker=mk, ms=2.3, mfc="white", label=f"{name}, training states")
        ax.set_ylim(-3, 100)
        ax.set_title(f"({'abcdef'[3 * i + j]}) {fname}, {cname}", loc="left", pad=2)
        if j == 0:
            ax.set_ylabel("States miscoordinated (%)")
    ax = axs[2, j]; tidy(ax)
    for case, ls, cname in (("der_rich", "-", "DER-rich"), ("moderate", "--", "moderate")):
        s = summ(fd, case)
        for kind, col, mk, name in (("cluster", BLUE, "o", "Clustered"), ("scalar", ORANGE, "s", "Scalar")):
            x, _, _, ung = sweep_series(s, kind)
            ax.plot(x, ung, color=col, ls=ls, marker=mk, ms=2.8, label=f"{name}, {cname}")
    ax.set_xlabel("Number of setting groups")
    ax.set_xticks([1, 2, 3, 4, 6, 8, 11])
    ax.set_ylim(-0.3, 8)
    ax.set_title(f"({'ghi'[j]}) {fname}: ungradable", loc="left", pad=2)
    if j == 0:
        ax.set_ylabel("Ungradable groups per partition")
hdl, lbl = axs[0, 0].get_legend_handles_labels()
fig.legend(hdl, lbl, loc="upper center", ncol=4, frameon=False, fontsize=7.3, handlelength=2.2,
           bbox_to_anchor=(0.5, 0.955), columnspacing=1.4)
h2, l2 = axs[2, 0].get_legend_handles_labels()
fig.legend(h2, l2, loc="lower center", ncol=4, frameon=False, fontsize=7.3, handlelength=2.2,
           bbox_to_anchor=(0.5, -0.02), columnspacing=1.4)
fig.savefig(os.path.join(OUT, "fig_group_sweep.pdf"), bbox_inches="tight")
plt.close(fig)


# ---------------- worst margin per unseen state (DER-rich, replication 0) ------
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True, gridspec_kw=dict(wspace=0.12))
styles = (("conventional", "Conventional (peak load, no DER)", ORANGE, "-"),
          ("scalar_11", "Scalar index, 11 levels", PURPLE, "-."),
          ("robust_K1", "Single robust group", INK2, ":"),
          ("proposed_knn_env", "Proposed, k-NN + envelope", BLUE, "-"))
for ax, (fd, fname) in zip(axs, FEEDERS):
    tidy(ax)
    m = rep0(fd, "der_rich")["margins"].item()
    for key, name, col, ls in styles:
        w = np.sort(np.nanmin(m[key], axis=1))
        ax.step(w, np.arange(1, len(w) + 1) / len(w) * 100, where="post", color=col, ls=ls, lw=1.3, label=name)
    ax.axvline(0.30, color=INK, lw=0.7, ls="--")
    ax.text(0.298, 8, "CTI", rotation=90, ha="right", va="bottom", fontsize=6.8, color=INK)
    ax.set_xlim(0.20, 0.36)
    ax.set_xlabel("Worst pair margin in the state (s)")
    ax.set_title(f"({'abc'[FEEDERS.index((fd, fname))]}) {fname}, DER-rich", loc="left", pad=2)
axs[0].set_ylabel("Unseen states (cumulative %)")
hm, lm = axs[0].get_legend_handles_labels()
fig.legend(hm, lm, loc="lower center", ncol=4, frameon=False, fontsize=7.3, handlelength=2.4,
           bbox_to_anchor=(0.5, -0.24), columnspacing=1.4)
fig.savefig(os.path.join(OUT, "fig_margins.pdf"), bbox_inches="tight")
plt.close(fig)


# ---------------- sensitivity: source strength and inverter current limit -----
XS = ((0.10, "_zs0.10"), (0.15, "_zs0.15"), (0.20, ""), (0.40, "_zs0.40"))
KF = ((1.2, "_kf1.2"), (1.5, "_kf1.5"), (2.0, ""))
if all(summ(fd, "der_rich", t) for fd, _ in FEEDERS for _, t in XS + KF):
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.45), sharey=True, gridspec_kw=dict(wspace=0.10))
    cols = {"ieee33": BLUE, "ieee69": ORANGE, "ieee123": AQUA}
    for ax, (grid, xlabel, title) in zip(axs, (
            (XS, "Source reactance $X_s$ (pu, 10 MVA base)", "(a) Grid strength ($k_f$ = 2)"),
            (KF, "Inverter fault-current limit $k_f$ (× rated)", "(b) Inverter current limit ($X_s$ = 0.20 pu)"))):
        tidy(ax)
        xs = [x for x, _ in grid]
        for fd, fname in FEEDERS:
            S = [summ(fd, "der_rich", t) for _, t in grid]
            conv = [x["schemes"]["conventional"]["affected"]["mean"] for x in S]
            prop = [x["schemes"]["proposed_knn_env"]["affected"]["mean"] for x in S]
            ax.plot(xs, conv, color=cols[fd], marker="o", ms=3.6, label=f"{fname}: conventional")
            ax.plot(xs, prop, color=cols[fd], marker="s", ms=3.0, ls="--", mfc="white", label=f"{fname}: proposed")
            for x, sx in zip(xs, S):                     # no scheme gradable: the single group fails too
                if sx["schemes"]["robust_K1"]["affected"]["mean"] > 99.0:
                    ax.plot([x], [100], marker="x", ms=7, color=cols[fd], mew=1.4, ls="none")
        ax.set_xticks(xs); ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylim(-3, 108)
    axs[0].text(0.16, 99, "× no setting can grade the\n   33- and 69-bus feeders", fontsize=6.8, color=INK2, va="top")
    axs[0].set_ylabel("Unseen states miscoordinated (%)")
    hs, ls_ = axs[1].get_legend_handles_labels()
    fig.legend(hs, ls_, loc="lower center", ncol=3, frameon=False, fontsize=7.2, handlelength=2.4,
               bbox_to_anchor=(0.5, -0.20), columnspacing=1.4)
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
print("figures written to", OUT)
