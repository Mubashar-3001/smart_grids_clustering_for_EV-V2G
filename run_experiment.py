#!/usr/bin/env python3
"""
Reproduces every number reported in the paper

  "Optimized Setting-Group Finding: Real-Data Clustering for Adaptive
   Overcurrent Protection in DER-Rich Smart Grids with EV/V2G Hubs"

    python run_experiment.py --feeder ieee33 --case moderate      # 10 replications
    python run_experiment.py --feeder ieee69 --case der_rich
    python run_experiment.py --feeder ieee123 --case der_rich --reps 1   # quick check
    python run_experiment.py --feeder ieee123 --case der_rich --zs 0.40 --no-sweep --tag _zs0.40

Requires numpy, scipy and scikit-learn. Reads ../data/ausgrid_c12_*.csv (real
Ausgrid customer data, Sydney, see ../data/prepare_ausgrid_data.py) and the
69-/123-bus CSV files built by ../data/prepare_feeders.py. Writes
results/summary_<feeder>_<case><tag>.json and results/rep0_<feeder>_<case><tag>.npz.
Modelling assumptions are tagged [A1]-[A7] as in the paper.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# =========================================================================
# 1. IEEE 33-bus feeder (Baran & Wu, 1989), 12.66 kV, Sbase = 10 MVA
# =========================================================================
VBASE_KV, SBASE_MVA = 12.66, 10.0
SBASE_KW = SBASE_MVA * 1e3
ZBASE = VBASE_KV ** 2 / SBASE_MVA
IBASE = SBASE_MVA * 1e6 / (np.sqrt(3) * VBASE_KV * 1e3)
ZS = 0.20j                      # grid Thevenin impedance, ~50 MVA fault level
NBUS, NREL, NPAIR, NHUB, N_TRUNK = 33, 9, 8, 6, 6

BRANCHES = np.array([
    [1, 2, .0922, .0470], [2, 3, .4930, .2511], [3, 4, .3660, .1864],
    [4, 5, .3811, .1941], [5, 6, .8190, .7070], [6, 7, .1872, .6188],
    [7, 8, .7114, .2351], [8, 9, 1.0300, .7400], [9, 10, 1.0440, .7400],
    [10, 11, .1966, .0650], [11, 12, .3744, .1238], [12, 13, 1.4680, 1.1550],
    [13, 14, .5416, .7129], [14, 15, .5910, .5260], [15, 16, .7463, .5450],
    [16, 17, 1.2890, 1.7210], [17, 18, .7320, .5740], [2, 19, .1640, .1565],
    [19, 20, 1.5042, 1.3554], [20, 21, .4095, .4784], [21, 22, .7089, .9373],
    [3, 23, .4512, .3083], [23, 24, .8980, .7091], [24, 25, .8960, .7011],
    [6, 26, .2030, .1034], [26, 27, .2842, .1447], [27, 28, 1.0590, .9337],
    [28, 29, .8042, .7006], [29, 30, .5075, .2585], [30, 31, .9744, .9630],
    [31, 32, .3105, .3619], [32, 33, .3410, .5302]])
LOADS = np.array([  # bus, P (kW), Q (kVAr): peak values
    [1, 0, 0], [2, 100, 60], [3, 90, 40], [4, 120, 80], [5, 60, 30], [6, 60, 20],
    [7, 200, 100], [8, 200, 100], [9, 60, 20], [10, 60, 20], [11, 45, 30],
    [12, 60, 35], [13, 60, 35], [14, 120, 80], [15, 60, 10], [16, 60, 20],
    [17, 60, 20], [18, 90, 40], [19, 90, 40], [20, 90, 40], [21, 90, 40],
    [22, 90, 40], [23, 90, 50], [24, 420, 200], [25, 420, 200], [26, 60, 25],
    [27, 60, 25], [28, 60, 20], [29, 120, 70], [30, 200, 600], [31, 150, 70],
    [32, 210, 100], [33, 60, 40]])
PEAK_KW = LOADS[:, 1].sum()                                  # 3715 kW

RELAYS = ["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3"]
RELAY_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (2, 19), (3, 23), (6, 26)]
# (primary, backup, zone-boundary fault bus), relay indices into RELAYS
PAIRS = [(1, 0, 3), (2, 1, 4), (3, 2, 5), (4, 3, 6), (5, 4, 7),
         (6, 0, 19), (7, 1, 23), (8, 4, 26)]
PAIR_NAMES = [f"{RELAYS[p]}/{RELAYS[b]}" for p, b, _ in PAIRS]
P_IDX = np.array([p for p, _, _ in PAIRS])
B_IDX = np.array([b for _, b, _ in PAIRS])
HUB_BUSES = [9, 14, 17, 21, 24, 30]


def _ybus():
    Y = np.zeros((NBUS, NBUS), complex)
    for f, t, r, x in BRANCHES:
        i, j, y = int(f) - 1, int(t) - 1, ZBASE / complex(r, x)
        Y[i, i] += y; Y[j, j] += y; Y[i, j] -= y; Y[j, i] -= y
    Y[0, 0] += 1.0 / ZS
    return Y


Y0 = _ybus()
Z_PRE = np.linalg.inv(Y0)                       # pre-fault (load) solve
Z_FLT = {}                                      # bolted 3-phase fault solves
for _, _, fb in PAIRS:
    Yf = Y0.copy(); Yf[fb - 1, fb - 1] += 1e6
    Z_FLT[fb] = np.linalg.inv(Yf)
_BR_Y = {(int(f), int(t)): ZBASE / complex(r, x) for f, t, r, x in BRANCHES}
REL_F = np.array([f - 1 for f, _ in RELAY_EDGES])
REL_T = np.array([t - 1 for _, t in RELAY_EDGES])
REL_Y = np.array([_BR_Y[e] for e in RELAY_EDGES])


# -------------------------------------------------------------------------
# Feeder library. 33-bus data above; 69- and 123-bus data are read from
# ../data (built by ../data/prepare_feeders.py). Each layout has six trunk
# DOCRs R1-R6 and three lateral-head DOCRs L1-L3; a lateral head is backed up
# by the nearest relay upstream of it, and each pair is checked for a bolted
# fault at the far end of the primary relay's branch. Hubs: one on each
# protected lateral, three on the trunk beyond R6 (same rule on all three feeders).
# -------------------------------------------------------------------------
FEEDER_LAYOUTS = {
    "ieee33": dict(vbase=12.66, relays=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (2, 19), (3, 23), (6, 26)],
                   hubs=[9, 14, 17, 21, 24, 30]),
    "ieee69": dict(vbase=12.66, relays=[(2, 3), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (4, 47), (8, 51), (9, 53)],
                   hubs=[12, 16, 21, 49, 52, 61]),
    # IEEE names; mapped to indices through feeder123_busmap.csv
    "ieee123": dict(vbase=4.16, relays=[("149", "1"), ("13", "152"), ("54", "57"), ("60", "160"), ("67", "72"),
                                        ("76", "86"), ("13", "18"), ("60", "62"), ("67", "97")],
                    hubs=["87", "91", "95", "50", "65", "101"]),
}
FEEDER = "ieee33"
BUS_NAME = {i: str(i) for i in range(1, NBUS + 1)}


def _derive_pairs(branches, relay_edges):
    parent = {int(t): int(f) for f, t, _, _ in branches}
    rel_at = {int(t): k for k, (f, t) in enumerate(relay_edges)}   # relay feeding bus t
    pairs = []
    for k, (f, t) in enumerate(relay_edges):
        u, back = int(f), None
        while u in parent:                    # walk upstream from the relay's from-bus
            if u in rel_at:
                back = rel_at[u]; break
            u = parent[u]
        if back is not None:
            pairs.append((k, back, int(t)))
    return pairs


def set_feeder(name, zs=0.20j):
    """Rebind all feeder-dependent globals (network, relays, pairs, hubs, Z)."""
    global FEEDER, VBASE_KV, ZBASE, IBASE, ZS, NBUS, BRANCHES, LOADS, PEAK_KW, RELAY_EDGES
    global PAIRS, PAIR_NAMES, P_IDX, B_IDX, HUB_BUSES, Y0, Z_PRE, Z_FLT, REL_F, REL_T, REL_Y, BUS_NAME
    global NREL, NPAIR, N_TRUNK, RELAYS
    lay = FEEDER_LAYOUTS[name]
    FEEDER, VBASE_KV, ZS = name, lay["vbase"], zs
    ZBASE = VBASE_KV ** 2 / SBASE_MVA
    IBASE = SBASE_MVA * 1e6 / (np.sqrt(3) * VBASE_KV * 1e3)
    if name == "ieee33":
        BRANCHES, LOADS = _BR33, _LD33
        BUS_NAME = {i: str(i) for i in range(1, 34)}
        idx = lambda b: int(b)
    else:
        n = name.replace("ieee", "")
        BRANCHES = np.loadtxt(os.path.join(DATA, f"feeder{n}_branches.csv"), delimiter=",", skiprows=1)
        LOADS = np.loadtxt(os.path.join(DATA, f"feeder{n}_loads.csv"), delimiter=",", skiprows=1)
        if name == "ieee123":
            rows = [l.strip().split(",") for l in open(os.path.join(DATA, "feeder123_busmap.csv")).readlines()[1:]]
            BUS_NAME = {int(i): b for i, b in rows}
            inv = {b: int(i) for i, b in rows}
            idx = lambda b: inv[str(b)]
        else:
            BUS_NAME = {i: str(i) for i in range(1, len(LOADS) + 1)}
            idx = lambda b: int(b)
    NBUS = len(LOADS)
    PEAK_KW = LOADS[:, 1].sum()
    RELAY_EDGES = [(idx(f), idx(t)) for f, t in lay["relays"]]
    N_TRUNK = lay.get("n_trunk", 6)
    NREL = len(RELAY_EDGES)
    RELAYS = [f"R{k + 1}" for k in range(N_TRUNK)] + [f"L{k + 1}" for k in range(NREL - N_TRUNK)]
    HUB_BUSES = [idx(b) for b in lay["hubs"]]
    PAIRS = _derive_pairs(BRANCHES, RELAY_EDGES)
    NPAIR = len(PAIRS)
    PAIR_NAMES = [f"{RELAYS[p]}/{RELAYS[b]}" for p, b, _ in PAIRS]
    P_IDX = np.array([p for p, _, _ in PAIRS]); B_IDX = np.array([b for _, b, _ in PAIRS])
    Y = np.zeros((NBUS, NBUS), complex)
    for f, t, r, x in BRANCHES:
        i, j, y = int(f) - 1, int(t) - 1, ZBASE / complex(r, x)
        Y[i, i] += y; Y[j, j] += y; Y[i, j] -= y; Y[j, i] -= y
    Y[0, 0] += 1.0 / ZS
    Y0 = Y
    Z_PRE = np.linalg.inv(Y0)
    Z_FLT = {}
    for _, _, fb in PAIRS:
        Yf = Y0.copy(); Yf[fb - 1, fb - 1] += 1e6
        Z_FLT[fb] = np.linalg.inv(Yf)
    bry = {(int(f), int(t)): ZBASE / complex(r, x) for f, t, r, x in BRANCHES}
    REL_F = np.array([f - 1 for f, _ in RELAY_EDGES])
    REL_T = np.array([t - 1 for _, t in RELAY_EDGES])
    REL_Y = np.array([bry[e] for e in RELAY_EDGES])


_BR33, _LD33 = BRANCHES.copy(), LOADS.copy()


def relay_flows(V):
    """Bus voltages V (NBUS, N) -> relay current magnitudes (N, NREL) in A and
    a mask that is True where active power flows in the relay's forward
    (downstream) direction."""
    I = (V[REL_F] - V[REL_T]) * REL_Y[:, None]
    fwd = np.real(V[REL_F] * np.conj(I)) > 0.0
    return (np.abs(I) * IBASE).T, fwd.T


# =========================================================================
# 2. Real data (Ausgrid customer 12, 366 days) and DER model
# =========================================================================
_pv = np.loadtxt(os.path.join(DATA, "ausgrid_c12_pv_pu.csv"), delimiter=",", skiprows=1)
DAY_MONTH, DAY_WD, PV_PU = _pv[:, 1].astype(int), _pv[:, 3].astype(int), _pv[:, 4:]
LOAD_SHAPE = np.loadtxt(os.path.join(DATA, "ausgrid_c12_load_shape.csv"),
                        delimiter=",", skiprows=1)
NDAYS = PV_PU.shape[0]

# [A3] hub = workplace EV site (ACN-sized, ~50 EVSEs) + PV canopy; sizes fixed
P_EV, N_EVSE_NOM, CAP_PER_EVSE = 7.2, 50, 3.75
HUB_SCALE = np.random.default_rng(2026).uniform(0.6, 1.4, NHUB)
N_EVSE = np.round(N_EVSE_NOM * HUB_SCALE).astype(int)
SITE_CAP = CAP_PER_EVSE * N_EVSE
SESS_WD, SESS_WE = 0.8, 0.25      # sessions per EVSE per weekday / weekend day
V2G_SHARE, V2G_PROB = 0.30, 0.35  # [A4] V2G-capable share; P(export | idle)
KF = 2.0                          # [A5] inverter fault-current limit (x rated)
PV_ONLINE = 0.02                  # PV inverters online above 2 % of rating
# [A3] session GMMs over (arrival h, duration h, energy kWh): weight, means, SDs
GMM_WD = [(0.60, (8.25, 8.5, 9.0), (1.0, 1.5, 3.0)),
          (0.25, (12.5, 3.5, 6.0), (2.0, 1.5, 2.0)),
          (0.15, (18.0, 4.0, 7.0), (1.5, 2.0, 2.5))]
GMM_WE = [(1.00, (13.0, 4.0, 7.0), (3.5, 2.0, 2.5))]
# DER cases: hub PV canopy nominal kW; rooftop PV at every load bus (x bus peak)
CASES = {"moderate": dict(hub_pv=300.0, roof=0.0), "der_rich": dict(hub_pv=600.0, roof=0.6)}
PV_RATED, ROOF = PV_NOM_DEFAULT = 300.0 * HUB_SCALE, 0.0


def set_case(name):
    global PV_RATED, ROOF
    PV_RATED, ROOF = CASES[name]["hub_pv"] * HUB_SCALE, CASES[name]["roof"]


def generate(n, seed):
    """n operating scenarios, each anchored to a real (day, half-hour) of the
    Ausgrid record: real PV output, real month/weekday, real load shape."""
    rng = np.random.default_rng(seed)
    day = rng.integers(0, NDAYS, n)
    slot = rng.integers(0, 48, n)
    hour = slot * 0.5 + rng.uniform(0.0, 0.5, n)
    weekend = DAY_WD[day] >= 5
    s_load = LOAD_SHAPE[DAY_MONTH[day] - 1, slot]        # [A2]
    pv = PV_PU[day, slot]                                # [A1]
    net, infeed, ev, v2g = (np.zeros((n, NHUB)) for _ in range(4))
    for i in range(n):
        gmm, rate = (GMM_WE, SESS_WE) if weekend[i] else (GMM_WD, SESS_WD)
        w = np.array([g[0] for g in gmm])
        for h in range(NHUB):
            m = rng.poisson(rate * N_EVSE[h])
            imp = exp_ = 0.0
            if m > 0:
                comp = rng.choice(len(gmm), size=m, p=w)
                mu = np.array([gmm[c][1] for c in comp])
                sd = np.array([gmm[c][2] for c in comp])
                x = mu + sd * rng.standard_normal((m, 3))
                arr = np.mod(x[:, 0], 24.0)
                dur = np.clip(x[:, 1], 0.25, 14.0)
                en = np.clip(x[:, 2], 1.0, 40.0)
                dt = np.mod(hour[i] - arr, 24.0)
                conn = dt < dur
                chg = conn & (dt < np.minimum(en / P_EV, dur))   # uncontrolled
                cap = rng.random(m) < V2G_SHARE
                expo = conn & ~chg & cap & (rng.random(m) < V2G_PROB)
                imp = min(P_EV * chg.sum(), SITE_CAP[h])
                exp_ = min(P_EV * expo.sum(), SITE_CAP[h])
            pv_kw = PV_RATED[h] * pv[i]
            net[i, h] = imp - exp_ - pv_kw               # + import, - export
            infeed[i, h] = KF * (exp_ + PV_RATED[h] * (pv[i] > PV_ONLINE))
            ev[i, h], v2g[i, h] = imp, exp_
    return dict(day=day, slot=slot, hour=hour, weekend=weekend, s_load=s_load,
                pv=pv, net=net, infeed=infeed, ev=ev, v2g=v2g)


def base_case():
    """Conventional design point: peak load, no DER."""
    z = np.zeros((1, NHUB))
    return dict(s_load=np.ones(1), pv=np.zeros(1), net=z, infeed=z.copy())


def currents(S):
    """Forward (directional) relay load currents and, per pair, primary and
    backup fault currents -- each relay sees its OWN current, so DER infeed
    between a primary and its backup is captured."""
    n = len(S["s_load"])
    pv = S["pv"]
    I = np.zeros((NBUS, n), complex)
    I[0] = 1.0 / ZS
    for b, p, q in LOADS:
        I[int(b) - 1] -= (p + 1j * q) / SBASE_KW * S["s_load"]
        I[int(b) - 1] += ROOF * p * pv / SBASE_KW               # rooftop PV
    for h, b in enumerate(HUB_BUSES):
        I[b - 1] -= S["net"][:, h] / SBASE_KW                   # unity-pf hub
    amps, fwd = relay_flows(Z_PRE @ I)
    # [A6] pickups are sized on load-current magnitude in either direction;
    # encroachment is checked on forward flow only (directional element)
    load = (amps, np.where(fwd, amps, 0.0))
    J = np.zeros((NBUS, n), complex)
    J[0] = 1.0 / ZS
    for b, p, _ in LOADS:
        J[int(b) - 1] += KF * ROOF * p * (pv > PV_ONLINE) / SBASE_KW
    for h, b in enumerate(HUB_BUSES):
        J[b - 1] += S["infeed"][:, h] / SBASE_KW                # [A5]
    prim, back = np.zeros((n, NPAIR)), np.zeros((n, NPAIR))
    for k, (p, b, fb) in enumerate(PAIRS):
        a, _ = relay_flows(Z_FLT[fb] @ J)
        prim[:, k], back[:, k] = a[:, p], a[:, b]
    return load, prim, back


def online_capacity(S):
    return S["infeed"].sum(1) / KF + ROOF * PEAK_KW * (S["pv"] > PV_ONLINE)


def cluster_features(S):
    """Offline operating-state descriptor (true per-hub states)."""
    return np.column_stack([S["net"], PEAK_KW * S["s_load"], online_capacity(S)])


def measurable_features(S):
    """Online inputs available to a DNSP: feeder-head net P, two zone
    aggregates, online inverter capacity (telemetry), time of day."""
    head = PEAK_KW * S["s_load"] * (1 - 0) + S["net"].sum(1) - ROOF * PEAK_KW * S["pv"]
    return np.column_stack([head, S["net"][:, :3].sum(1), S["net"][:, 3:].sum(1),
                            online_capacity(S), np.sin(2 * np.pi * S["hour"] / 24),
                            np.cos(2 * np.pi * S["hour"] / 24)])


# =========================================================================
# 3. DOCR coordination: IEC very inverse, CTI, pickups, robust WOA (+LP check)
# =========================================================================
A_IEC, CTI, T_MIN, TMS_LO, TMS_HI = 13.5, 0.30, 0.10, 0.05, 1.10
CTI_DESIGN = 0.31               # 10 ms design margin; events are counted at CTI
OLF, PK_MIN = 1.25, 50.0
W_CTI, W_BLIND, T_BLIND = 500.0, 800.0, 5.0
N_AGENTS, N_ITERS = 40, 400


def op_time(tms, I, Ip):
    M = I / Ip
    with np.errstate(divide="ignore", invalid="ignore"):
        t = tms * A_IEC / (M - 1.0)
    return np.where(M > 1.0 + 1e-9, np.maximum(t, T_MIN), np.inf)


def pickups(load_rows):
    if isinstance(load_rows, tuple):
        load_rows = load_rows[0]
    return np.maximum(OLF * np.max(np.atleast_2d(load_rows), axis=0), PK_MIN)


def sel(L, idx):
    """Row-select a (magnitude, forward) load pair."""
    return (L[0][idx], L[1][idx])


def fitness_pop(T, prim, back, pk):
    """Robust (minimax) fitness of P candidate TMS vectors over S scenarios."""
    tp = op_time(T[:, P_IDX][:, None, :], prim[None], pk[P_IDX])     # (P,S,NPAIR)
    tb = op_time(T[:, B_IDX][:, None, :], back[None], pk[B_IDX])
    bp, bb = ~np.isfinite(tp), ~np.isfinite(tb)
    with np.errstate(invalid="ignore"):
        short = np.where(bp | bb, 0.0, np.maximum(CTI_DESIGN - (tb - tp), 0.0))
    F = (np.where(bp, T_BLIND, tp) + W_BLIND * (bp.astype(float) + bb) + W_CTI * short).sum(2)
    return F.max(1) + 0.01 * F.mean(1)


def woa(fit, dim, n_agents=N_AGENTS, iters=N_ITERS, seed=0):
    """Whale Optimization Algorithm (Mirjalili & Lewis, 2016) with vector
    coefficients A = 2a*r1 - a, C = 2*r2 (r1, r2 ~ U(0,1)^d); encircle the
    leader if ||A|| < 1, else follow a random whale; logarithmic spiral with
    probability 0.5; synchronous update."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(TMS_LO, TMS_HI, (n_agents, dim))
    f = fit(X); i = int(np.argmin(f))
    best, fbest, hist = X[i].copy(), f[i], [f[i]]
    for t in range(iters):
        a = 2.0 - 2.0 * t / iters
        A = 2 * a * rng.random((n_agents, dim)) - a
        C = 2 * rng.random((n_agents, dim))
        p = rng.random(n_agents)[:, None]
        l = rng.uniform(-1.0, 1.0, (n_agents, dim))
        Xr = X[rng.integers(0, n_agents, n_agents)]
        small = (np.linalg.norm(A, axis=1) < 1.0)[:, None]
        enc = best - A * np.abs(C * best - X)
        exr = Xr - A * np.abs(C * Xr - X)
        spi = np.abs(best - X) * np.exp(l) * np.cos(2 * np.pi * l) + best
        X = np.clip(np.where(p < 0.5, np.where(small, enc, exr), spi), TMS_LO, TMS_HI)
        f = fit(X); i = int(np.argmin(f))
        if f[i] < fbest:
            best, fbest = X[i].copy(), f[i]
        hist.append(fbest)
    return best, fbest, np.array(hist)


def lp_reference(prim, back, pk, return_x=False):
    """Exact LP of the robust TMS problem, Eq. (5), solved with HiGHS.
    With pickups fixed, t = max(k*TMS, T_MIN) is convex piecewise linear in
    TMS, so an epigraph variable u_sk >= k_sk*TMS_p, u_sk >= T_MIN per state s
    and pair k makes the minimax objective linear. The grading constraint
    max(kb*TMS_q, T_MIN) >= max(kp*TMS_p, T_MIN) + CTI_DESIGN splits exactly
    into kb*TMS_q - kp*TMS_p >= CTI_DESIGN and kb*TMS_q >= T_MIN + CTI_DESIGN.
    The LP is therefore the global optimum of the penalty-free problem, and
    its infeasibility certifies that no TMS vector can grade the group.
    Returns the robust fitness of the LP solution (np.inf if infeasible)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    S, n, m = prim.shape[0], NREL, len(P_IDX)
    Mp, Mb = prim / pk[P_IDX], back / pk[B_IDX]
    if np.any(Mp <= 1.0 + 1e-9) or np.any(Mb <= 1.0 + 1e-9):
        return (np.inf, None) if return_x else np.inf          # a relay cannot see the fault
    kp, kb = A_IEC / (Mp - 1.0), A_IEC / (Mb - 1.0)
    nv = n + 1 + S * m                                         # [TMS | z | u_sk]
    iz, iu = n, n + 1 + np.arange(S * m).reshape(S, m)
    c = np.zeros(nv); c[iz] = 1.0; c[iu.ravel()] = 0.01 / S
    ss, kk = np.meshgrid(np.arange(S), np.arange(m), indexing="ij")
    ss, kk = ss.ravel(), kk.ravel(); nr = S * m
    # (i)  kp*TMS_p - u <= 0                 rows 0 .. nr-1
    r1 = np.arange(nr)
    rows = [r1, r1]; cols = [P_IDX[kk], iu[ss, kk]]; vals = [kp[ss, kk], -np.ones(nr)]
    # (ii) kp*TMS_p - kb*TMS_q <= -CTI_D     rows nr .. 2nr-1
    r2 = nr + np.arange(nr)
    rows += [r2, r2]; cols += [P_IDX[kk], B_IDX[kk]]; vals += [kp[ss, kk], -kb[ss, kk]]
    # (iii) sum_k u_sk - z <= 0              rows 2nr .. 2nr+S-1
    r3 = 2 * nr + ss
    rows += [r3, 2 * nr + np.arange(S)]; cols += [iu[ss, kk], np.full(S, iz)]; vals += [np.ones(nr), -np.ones(S)]
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * nr + S, nv)).tocsr()
    b = np.concatenate([np.zeros(nr), np.full(nr, -CTI_DESIGN), np.zeros(S)])
    # (iv) kb*TMS_q >= T_MIN + CTI_D as a lower bound on each backup TMS
    lo = np.full(n, TMS_LO)
    for k in range(m):
        q = B_IDX[k]; lo[q] = max(lo[q], (T_MIN + CTI_DESIGN) / kb[:, k].min())
    if np.any(lo > TMS_HI + 1e-12):
        return (np.inf, None) if return_x else np.inf
    bounds = [(lo[i], TMS_HI) for i in range(n)] + [(0, None)] + [(T_MIN, None)] * nr
    sol = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if sol.status != 0:
        return (np.inf, None) if return_x else np.inf
    x = np.clip(sol.x[:n], TMS_LO, TMS_HI)
    f = float(fitness_pop(x[None], prim, back, pk)[0])
    return (f, x) if return_x else f


def min_tms(prim, back, pk):
    """Least feasible TMS vector by a bottom-up recursion (each relay after all
    relays it backs up). The feasible set of Eq. (2) is closed under the
    componentwise minimum and the objective of Eq. (4) is nondecreasing in
    every TMS, so this vector is LP-optimal; None if the group is ungradable.
    Used to cross-check the LP (verify_lp.py) and, toolbox-free, in MATLAB."""
    Mp, Mb = prim / pk[P_IDX], back / pk[B_IDX]
    if np.any(Mp <= 1.0 + 1e-9) or np.any(Mb <= 1.0 + 1e-9):
        return None
    kp, kb = A_IEC / (Mp - 1.0), A_IEC / (Mb - 1.0)
    T, done = np.full(NREL, np.nan), np.zeros(NREL, bool)
    while not done.all():
        for r in np.where(~done)[0]:
            ks = np.where(B_IDX == r)[0]
            if all(done[P_IDX[k]] for k in ks):
                t = TMS_LO
                for k in ks:
                    tp = np.maximum(kp[:, k] * T[P_IDX[k]], T_MIN)
                    t = max(t, float(np.max((tp + CTI_DESIGN) / kb[:, k])))
                if t > TMS_HI + 1e-12:
                    return None
                T[r], done[r] = t, True
    return T


def gradable(prim, back, pk):
    """LP certificate: can the relay chain be graded at all with these pickups?"""
    return bool(np.isfinite(lp_reference(prim, back, pk)))


def optimise(prim, back, pk, seed):
    T, f, h = woa(lambda X: fitness_pop(X, prim, back, pk), NREL, seed=seed)
    return T, h


def evaluate(T_rows, PK_rows, load, prim, back):
    """Miscoordination events on scenarios given per-scenario settings."""
    tp = op_time(T_rows[:, P_IDX], prim, PK_rows[:, P_IDX])
    tb = op_time(T_rows[:, B_IDX], back, PK_rows[:, B_IDX])
    blind = ~np.isfinite(tp) | ~np.isfinite(tb)
    with np.errstate(invalid="ignore"):
        marg = np.where(blind, np.nan, tb - tp)
    cti = ~blind & (marg < CTI - 1e-9)
    enc = load[1] > PK_rows
    per = cti.sum(1) + blind.sum(1) + enc.sum(1)
    return dict(total=int(per.sum()), cti=int(cti.sum()), blind=int(blind.sum()),
                enc=int(enc.sum()), affected=float(100.0 * np.mean(per > 0)),
                t_prim=float(np.nanmean(np.where(np.isfinite(tp), tp, np.nan))),
                t_back=float(np.nanmean(np.where(np.isfinite(tb), tb, np.nan))),
                pk_trunk=float(PK_rows[:, :N_TRUNK].mean()),
                min_margin=float(np.nanmin(marg)) if np.isfinite(marg).any() else float('nan'),
                severe=int(np.sum(marg < 0.2)), margins=marg)


# =========================================================================
# 4. Clustering, validity selection, classifier, schemes
# =========================================================================
def stdz(Xtr, X):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def cluster(Xs, method, K, seed=0):
    if method == "kmeans":
        return KMeans(K, n_init=10, random_state=seed).fit_predict(Xs)
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xs)


def validity(Xs, Ks=range(2, 9)):
    rows = []
    for K in Ks:
        for m in ("kmeans", "ward"):
            lab = cluster(Xs, m, K)
            rows.append(dict(method=m, K=K, sil=float(silhouette_score(Xs, lab)),
                             db=float(davies_bouldin_score(Xs, lab))))
    return rows, sorted(rows, key=lambda r: (-r["sil"], r["db"]))[0]


def certified_selection(Xs, vrows, load, prim, back):
    """Optimized setting-group finding. Candidate partitions (method, K) are
    visited in order of decreasing silhouette (ties: lower Davies-Bouldin);
    the first one whose every group passes the exact LP gradability
    certificate is returned. Returns (row, labels, number of candidates
    examined); row is None, and the labels are all zero (K = 1, the single
    robust group), if no candidate is fully gradable."""
    order = sorted(vrows, key=lambda r: (-r["sil"], r["db"]))
    for i, c in enumerate(order):
        lab = cluster(Xs, c["method"], c["K"])
        if all(gradable(prim[lab == g], back[lab == g], pickups(sel(load, lab == g))) for g in range(c["K"])):
            return c, lab, i + 1
    return None, np.zeros(len(Xs), int), len(order)


def group_settings(labels, K, load, prim, back, seed):
    T, PK, H = np.zeros((K, NREL)), np.zeros((K, NREL)), []
    for c in range(K):
        idx = labels == c
        PK[c] = pickups(sel(load, idx))
        T[c], h = optimise(prim[idx], back[idx], PK[c], seed + c)
        H.append(h)
    return T, PK, H


def nearest(C, X):
    return np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)


def scalar_groups(nb, Str, Ste, Ltr, Ptr, Btr, seed):
    """Single aggregate-DER index rho = sum|hub net kW| (min-max over training),
    nb equally spaced bins, nearest-bin lookup (as in scalar-penetration maps)."""
    r_tr = np.abs(Str["net"]).sum(1); lo, hi = r_tr.min(), r_tr.max()
    r_tr = (r_tr - lo) / (hi - lo)
    r_te = np.clip((np.abs(Ste["net"]).sum(1) - lo) / (hi - lo), 0.0, 1.0)
    cen = np.linspace(0.0, 1.0, nb)
    b_tr = np.argmin(np.abs(r_tr[:, None] - cen[None]), 1)
    T, PK = np.zeros((nb, NREL)), np.zeros((nb, NREL))
    for j in range(nb):
        idx = np.where(b_tr == j)[0]
        if len(idx) < 10:
            idx = np.argsort(np.abs(r_tr - cen[j]))[:10]
        PK[j] = pickups(sel(Ltr, idx)); T[j], _ = optimise(Ptr[idx], Btr[idx], PK[j], seed + j)
    return T, PK, b_tr, np.argmin(np.abs(r_te[:, None] - cen[None]), 1)


K_SWEEP, NB_SWEEP = (2, 3, 4, 5, 6, 8), (2, 3, 4, 6, 8, 11)
DO_SWEEP = True


def run_rep(r, n_train, n_test, detail=False):
    t0 = time.time()
    Str, Ste = generate(n_train, 1000 + r), generate(n_test, 5000 + r)
    Ltr, Ptr, Btr = currents(Str)
    Lte, Pte, Bte = currents(Ste)
    out, seed = {}, 10000 + 100 * r

    def scheme(name, groups, T_rows, PK_rows, extra=None):
        e = evaluate(T_rows, PK_rows, Lte, Pte, Bte); e["groups"] = groups
        if extra:
            e.update(extra)
        out[name] = e

    # (a) conventional: one group designed at the peak-load, no-DER base case
    Lb, Pb, Bb = currents(base_case())
    pk0 = pickups(Lb); T0, _ = optimise(Pb, Bb, pk0, seed + 1)
    scheme("conventional", 1, np.tile(T0, (n_test, 1)), np.tile(pk0, (n_test, 1)))

    # (b) single robust group over all training scenarios (K = 1)
    pk1 = pickups(Ltr); T1, _ = optimise(Ptr, Btr, pk1, seed + 2)
    scheme("robust_K1", 1, np.tile(T1, (n_test, 1)), np.tile(pk1, (n_test, 1)))

    # (c) proposed: optimized setting-group finding (validity-ranked partitions,
    #     first fully LP-gradable one; K = 1 if none), and the validity-only
    #     baseline (silhouette-best partition without the certificate)
    Xtr, Xte = cluster_features(Str), cluster_features(Ste)
    Xs_tr, Xs_te = stdz(Xtr, Xtr), stdz(Xtr, Xte)
    vrows, best = validity(Xs_tr)
    chosen, lab, n_checked = certified_selection(Xs_tr, vrows, Ltr, Ptr, Btr)
    lab_v = cluster(Xs_tr, best["method"], best["K"])
    ungr_v = int(sum(not gradable(Ptr[lab_v == c], Btr[lab_v == c], pickups(sel(Ltr, lab_v == c)))
                     for c in range(best["K"])))
    if chosen is None:
        K, meth, sil_c, db_c = 1, "none", float("nan"), float("nan")
        Tk, PKk, Hk = T1[None].copy(), pk1[None].copy(), []
        cent = Xs_tr.mean(0)[None]
    else:
        K, meth, sil_c, db_c = chosen["K"], chosen["method"], chosen["sil"], chosen["db"]
        cent = np.array([Xs_tr[lab == c].mean(0) for c in range(K)])
        Tk, PKk, Hk = group_settings(lab, K, Ltr, Ptr, Btr, seed + 10)
    a_state = nearest(cent, Xs_te)
    if chosen is not None and chosen["K"] == best["K"] and chosen["method"] == best["method"]:
        scheme("validity_only", K, Tk[a_state], PKk[a_state])
    else:
        cv = np.array([Xs_tr[lab_v == c].mean(0) for c in range(best["K"])])
        Tv, PKv, _ = group_settings(lab_v, best["K"], Ltr, Ptr, Btr, seed + 10)
        a_v = nearest(cv, Xs_te)
        scheme("validity_only", best["K"], Tv[a_v], PKv[a_v])
    scheme("proposed_state", K, Tk[a_state], PKk[a_state])
    Mtr, Mte = measurable_features(Str), measurable_features(Ste)
    if K > 1:
        knn = KNeighborsClassifier(n_neighbors=5).fit(stdz(Mtr, Mtr), lab)
        t_c = time.perf_counter(); a_knn = knn.predict(stdz(Mtr, Mte)); t_c = time.perf_counter() - t_c
    else:
        a_knn, t_c = np.zeros(n_test, int), float("nan")
    acc = float(np.mean(a_knn == a_state))
    scheme("proposed_knn", K, Tk[a_knn], PKk[a_knn], dict(acc=acc))
    env = np.any(Lte[0] > 1.05 * PKk[a_knn] / OLF, axis=1)     # envelope check
    T_e, PK_e = Tk[a_knn].copy(), PKk[a_knn].copy()
    T_e[env], PK_e[env] = T1, pk1                               # fallback to K=1
    scheme("proposed_knn_env", K + (K > 1), T_e, PK_e, dict(fallback_pct=float(100 * env.mean())))
    # sensitivity in the PV-export regime (cluster with the highest mean PV)
    pvc = int(np.argmax([Str["pv"][lab == c].mean() for c in range(K)]))
    in_pv = a_state == pvc
    pk_red_pv = float(100 * (1 - PKk[pvc, :N_TRUNK].mean() / pk1[:N_TRUNK].mean()))
    pk_red_regimes = [float(100 * (1 - PKk[c, :N_TRUNK].mean() / pk1[:N_TRUNK].mean())) for c in range(K)]

    # (d) scalar aggregate-DER index with the prior-study resolution and with K bins
    for nb, name in ((11, "scalar_11"), (K, "scalar_K")):
        Ts, PKs, _, bte = scalar_groups(nb, Str, Ste, Ltr, Ptr, Btr, seed + 40 + 20 * (nb == K))
        scheme(name, nb, Ts[bte], PKs[bte])

    # (e) group-count sweep: in-sample vs out-of-sample (clustering and scalar bins)
    sweep = []
    for Kx in (K_SWEEP if DO_SWEEP else ()):
        lx = cluster(Xs_tr, "kmeans", Kx)
        cx = np.array([Xs_tr[lx == c].mean(0) for c in range(Kx)])
        Tx, PKx, _ = group_settings(lx, Kx, Ltr, Ptr, Btr, seed + 300 + 10 * Kx)
        ax = nearest(cx, Xs_te)
        e_in, e_out = evaluate(Tx[lx], PKx[lx], Ltr, Ptr, Btr), evaluate(Tx[ax], PKx[ax], Lte, Pte, Bte)
        ungr = sum(not gradable(Ptr[lx == c], Btr[lx == c], PKx[c]) for c in range(Kx))
        sweep.append(dict(kind="cluster", n=Kx, ungradable=ungr, in_aff=e_in["affected"], out_aff=e_out["affected"],
                          out_total=e_out["total"], pk_trunk=e_out["pk_trunk"], t_prim=e_out["t_prim"]))
    for nb in (NB_SWEEP if DO_SWEEP else ()):
        Ts, PKs, btr, bte = scalar_groups(nb, Str, Ste, Ltr, Ptr, Btr, seed + 600 + 20 * nb)
        e_in, e_out = evaluate(Ts[btr], PKs[btr], Ltr, Ptr, Btr), evaluate(Ts[bte], PKs[bte], Lte, Pte, Bte)
        ungr = sum(not gradable(Ptr[btr == j], Btr[btr == j], PKs[j]) for j in range(nb) if np.any(btr == j))
        sweep.append(dict(kind="scalar", n=nb, ungradable=ungr, in_aff=e_in["affected"], out_aff=e_out["affected"],
                          out_total=e_out["total"], pk_trunk=e_out["pk_trunk"], t_prim=e_out["t_prim"]))

    # WOA vs the exact LP optimum (relative gap in robust fitness, >= 0)
    # and the same groups with LP-optimal TMS (does the optimiser change the outcome?)
    gaps, T_lp = [], Tk.copy()
    for c in range(K):
        idx = lab == c
        fw = float(fitness_pop(Tk[c][None], Ptr[idx], Btr[idx], PKk[c])[0])
        fl, xl = lp_reference(Ptr[idx], Btr[idx], PKk[c], return_x=True)
        gaps.append(100.0 * (fw - fl) / fl if np.isfinite(fl) else float('nan'))
        if xl is not None:
            T_lp[c] = xl
    fw = float(fitness_pop(T1[None], Ptr, Btr, pk1)[0])
    fl = lp_reference(Ptr, Btr, pk1)
    gaps.append(100.0 * (fw - fl) / fl if np.isfinite(fl) else float('nan'))
    e_lp = evaluate(T_lp[a_state], PKk[a_state], Lte, Pte, Bte)
    lp_check = dict(affected=e_lp["affected"], min_margin=e_lp["min_margin"], t_prim=e_lp["t_prim"],
                    t_prim_woa=out["proposed_state"]["t_prim"])

    ins = evaluate(Tk[lab], PKk[lab], Ltr, Ptr, Btr)
    conv_it = [int(np.argmax(h <= h[-1] * (1 + 1e-3))) for h in Hk] or [0]
    meta = dict(K=K, method=meth, sil=sil_c, db=db_c, sizes=np.bincount(lab).tolist(),
                K_validity=best["K"], method_validity=best["method"], sil_validity=best["sil"],
                ungradable_validity=ungr_v, n_checked=n_checked,
                pv_cluster=pvc, pv_cluster_share_test=float(100 * in_pv.mean()),
                pk_red_pv_pct=pk_red_pv, pk_red_regimes=pk_red_regimes, insample_total=ins["total"], woa_lp_gap_pct=gaps,
                lp_check=lp_check,
                conv_iters=conv_it, knn_ms_per_sample=1e3 * t_c / n_test,
                t_rep_s=time.time() - t0, sweep=sweep,
                fc_spread_pct=(100 * (Pte.max(0) / Pte.min(0) - 1)).round(2).tolist())
    res = dict(meta=meta, schemes={k: {kk: vv for kk, vv in v.items() if kk != "margins"} for k, v in out.items()})
    det = None
    if detail:
        det = dict(validity=vrows, lab=lab, lab_validity=lab_v, hour_tr=Str["hour"], s_load_tr=Str["s_load"],
                   pv_tr=Str["pv"], net_tr=Str["net"], ev_tr=Str["ev"], v2g_tr=Str["v2g"],
                   weekend_tr=Str["weekend"], Tk=Tk, PKk=PKk, T1=T1, pk1=pk1, T0=T0, pk0=pk0,
                   conv=[h.tolist() for h in Hk], pair_names=PAIR_NAMES,
                   margins={k: v["margins"] for k, v in out.items()})
    return res, det


def summarise(reps):
    agg = {}
    for n in reps[0]["schemes"]:
        agg[n] = {}
        for key in ("total", "cti", "blind", "enc", "affected", "t_prim", "t_back", "pk_trunk",
                    "min_margin", "severe", "groups", "acc", "fallback_pct"):
            vals = [rr["schemes"][n].get(key) for rr in reps]
            if all(v is not None for v in vals):
                v = np.array(vals, float)
                agg[n][key] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                   min=float(v.min()), max=float(v.max()))
    sw = {}
    for d in reps[0]["meta"]["sweep"]:
        key = f'{d["kind"]}_{d["n"]}'
        rows = [next(x for x in rr["meta"]["sweep"] if x["kind"] == d["kind"] and x["n"] == d["n"]) for rr in reps]
        sw[key] = {m: dict(mean=float(np.mean([x[m] for x in rows])), std=float(np.std([x[m] for x in rows], ddof=1)) if len(rows) > 1 else 0.0)
                   for m in ("ungradable", "in_aff", "out_aff", "out_total", "pk_trunk", "t_prim")}
    g = np.concatenate([rr["meta"]["woa_lp_gap_pct"] for rr in reps])
    m = lambda k: float(np.mean([rr["meta"][k] for rr in reps]))
    g = g[np.isfinite(g)] if np.isfinite(g).any() else np.array([np.nan])
    mn = lambda k: float(np.nanmean([rr["meta"][k] for rr in reps])) if np.isfinite([rr["meta"][k] for rr in reps]).any() else float("nan")
    return dict(schemes=agg, sweep=sw, K_selected=[rr["meta"]["K"] for rr in reps],
                methods=[rr["meta"]["method"] for rr in reps],
                K_validity=[rr["meta"]["K_validity"] for rr in reps],
                methods_validity=[rr["meta"]["method_validity"] for rr in reps],
                ungradable_validity=[rr["meta"]["ungradable_validity"] for rr in reps],
                n_checked=[rr["meta"]["n_checked"] for rr in reps],
                sil_validity_mean=m("sil_validity"),
                sil_mean=mn("sil"), db_mean=mn("db"), pk_red_pv_pct=m("pk_red_pv_pct"),
                pk_red_regime_max=float(np.mean([max(rr["meta"]["pk_red_regimes"]) for rr in reps])),
                pv_cluster_share_test=m("pv_cluster_share_test"),
                insample_total=[rr["meta"]["insample_total"] for rr in reps],
                woa_lp_gap_pct=dict(min=float(g.min()), max=float(g.max()), mean=float(g.mean())),
                lp_check={k: float(np.mean([rr["meta"]["lp_check"][k] for rr in reps]))
                          for k in ("affected", "min_margin", "t_prim", "t_prim_woa")},
                lp_check_min_margin=float(min(rr["meta"]["lp_check"]["min_margin"] for rr in reps)),
                conv_iters_max=int(max(max(rr["meta"]["conv_iters"]) for rr in reps)),
                knn_ms_per_sample=mn("knn_ms_per_sample"), t_rep_s=m("t_rep_s"),
                fc_spread_pct_max=float(max(max(rr["meta"]["fc_spread_pct"]) for rr in reps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeder", choices=sorted(FEEDER_LAYOUTS), default="ieee33")
    ap.add_argument("--case", choices=sorted(CASES), default="moderate")
    ap.add_argument("--zs", type=float, default=0.20, help="source reactance (pu, 10 MVA base)")
    ap.add_argument("--kf", type=float, default=2.0, help="inverter fault-current limit (x rated)")
    ap.add_argument("--no-sweep", action="store_true", help="skip the group-count sweep")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--train", type=int, default=480)
    ap.add_argument("--test", type=int, default=480)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    global KF, DO_SWEEP
    KF, DO_SWEEP = a.kf, not a.no_sweep
    set_feeder(a.feeder, zs=1j * a.zs)
    set_case(a.case)
    stem = f"{a.feeder}_{a.case}{a.tag}"
    os.makedirs(a.out, exist_ok=True)
    print(f"[{a.feeder} | {a.case} | Zs=j{a.zs:.2f} pu | kf={KF}] peak={PEAK_KW:.0f} kW pairs={PAIR_NAMES} EVSE={N_EVSE.tolist()} hub PV kW={np.round(PV_RATED).tolist()} "
          f"site cap kW={np.round(SITE_CAP).tolist()} rooftop PV={ROOF:.0%} of bus peak")
    reps = []
    for r in range(a.reps):
        res, det = run_rep(r, a.train, a.test, detail=(r == 0))
        reps.append(res)
        mt, s = res["meta"], res["schemes"]
        print(f"rep {r}: K={mt['K']} ({mt['method']}, sil={mt['sil']:.3f}; validity-best K={mt['K_validity']} "
              f"{mt['method_validity']} with {mt['ungradable_validity']} ungradable, {mt['n_checked']} checked) sizes={mt['sizes']} | " +
              " ".join(f"{k}={v['total']}" for k, v in s.items()) +
              f" | acc={s['proposed_knn']['acc']:.3f} gap%={np.round(mt['woa_lp_gap_pct'], 2).tolist()} | {mt['t_rep_s']:.0f}s",
              flush=True)
        if det is not None:
            np.savez(os.path.join(a.out, f"rep0_{stem}.npz"),
                     **{k: (np.array(v, dtype=object) if isinstance(v, (list, dict)) else v) for k, v in det.items()})
    summ = summarise(reps)
    summ["config"] = dict(feeder=a.feeder, zs=a.zs, kf=KF, case=a.case, reps=a.reps, train=a.train, test=a.test, N_EVSE=N_EVSE.tolist(),
                          PV_RATED=PV_RATED.round(1).tolist(), SITE_CAP=SITE_CAP.round(1).tolist(), ROOF=ROOF)
    summ["reps"] = reps
    with open(os.path.join(a.out, f"summary_{stem}.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print("\n%-18s %6s %18s %16s %16s %12s %10s" % ("scheme", "groups", "test events", "affected %", "t_prim (s)", "trunk pk (A)", "min marg"))
    for n, v in summ["schemes"].items():
        print("%-18s %6.1f %9.1f +/- %5.1f %8.2f +/- %5.2f %7.4f +/- %.4f %8.1f %10.3f" % (
            n, v["groups"]["mean"], v["total"]["mean"], v["total"]["std"], v["affected"]["mean"],
            v["affected"]["std"], v["t_prim"]["mean"], v["t_prim"]["std"], v["pk_trunk"]["mean"],
            v["min_margin"]["min"]))
    print("K selected:", summ["K_selected"], summ["methods"])
    print("validity-best K:", summ["K_validity"], summ["methods_validity"], " ungradable groups in it:", summ["ungradable_validity"],
          " candidates checked:", summ["n_checked"])
    print("in-sample events (proposed):", summ["insample_total"], " WOA-LP gap %:", summ["woa_lp_gap_pct"])
    print("same groups with LP-optimal TMS (mean over reps):", {k: round(v, 4) for k, v in summ["lp_check"].items()},
          "worst margin %.3f s" % summ["lp_check_min_margin"])
    print("trunk pickup reduction in PV-export regime vs K=1: %.1f %% (regime share %.1f %%)" % (
        summ["pk_red_pv_pct"], summ["pv_cluster_share_test"]))
    print("sweep (out-of-sample affected %, mean):",
          {k: round(v["out_aff"]["mean"], 1) for k, v in summ["sweep"].items()})
    print("sweep (ungradable groups, mean):", {k: round(v["ungradable"]["mean"], 2) for k, v in summ["sweep"].items()})
    print("sweep (in-sample affected %, mean):",
          {k: round(v["in_aff"]["mean"], 1) for k, v in summ["sweep"].items()})


if __name__ == "__main__":
    main()
