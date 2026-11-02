#!/usr/bin/env python3
"""
Build the balanced positive-sequence feeder models used by both the Python
and MATLAB pipelines, and write them as plain CSV files next to this script.

  IEEE 33-bus : Baran & Wu (1989), tabulated in run_experiment.py / here.
  IEEE 69-bus : MATPOWER case69.m (Baran & Wu 1989; Das 2008), r/x in ohm.
                https://github.com/MATPOWER/matpower  (data/case69.m)
  IEEE 123-bus: IEEE PES 123-node test feeder, OpenDSS files from
                https://github.com/dss-extensions/electricdss-tst
                (Version8/Distrib/IEEETestCases/123Bus + IEEELineCodes.DSS)

123-bus reduction to a balanced positive-sequence model [A7]:
  * three-phase line codes: z1 = mean(self) - mean(mutual) of the phase matrix;
    two-phase: z1 = self - mutual; single-phase: z = self impedance;
  * voltage regulators are ideal (their in/out nodes merged), closed switches
    are 1 mOhm branches, the normally open ties Sw7 and Sw8 are left open;
  * the 150 kVA XFM-1 to bus 610 carries no load and is omitted;
  * all phase loads at a bus are summed; shunt capacitors enter as negative Q.

Usage:
  python prepare_feeders.py <path/to/case69.m> <path/to/123Bus folder> <path/to/IEEELineCodes.DSS>
Outputs: feeder69_branches.csv, feeder69_loads.csv,
         feeder123_branches.csv, feeder123_loads.csv, feeder123_busmap.csv
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_case69(path):
    txt = open(path).read()

    def block(name):
        m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % name, txt, re.S)
        rows = []
        for ln in m.group(1).splitlines():
            ln = ln.split("%")[0].strip().rstrip(";")
            if ln:
                rows.append([float(v) for v in ln.split()])
        return np.array(rows)
    bus, br = block("bus"), block("branch")
    branches = br[:, :4]                       # f, t, r (ohm), x (ohm)
    loads = bus[:, [0, 2, 3]]                  # bus, P kW, Q kVAr
    return branches, loads


def parse_linecodes(path):
    codes, cur = {}, None
    for raw in open(path):
        ln = raw.split("!")[0].strip()
        m = re.match(r"New linecode\.(\S+)\s+nphases=(\d)", ln, re.I)
        if m:
            cur = m.group(1); codes[cur] = {"n": int(m.group(2))}; continue
        for key in ("rmatrix", "xmatrix"):
            mm = re.search(key + r"\s*=\s*[\[\(](.*?)[\]\)]", ln, re.I)
            if mm and cur:
                vals = [float(v) for v in mm.group(1).replace("|", " ").split()]
                codes[cur][key[0]] = vals
    out = {}
    for k, c in codes.items():
        n = c["n"]
        M = {}
        for key in ("r", "x"):
            v = c[key]; A = np.zeros((n, n)); idx = 0
            for i in range(n):
                for j in range(i + 1):
                    A[i, j] = A[j, i] = v[idx]; idx += 1
            M[key] = A
        Z = M["r"] + 1j * M["x"]
        if n == 1:
            z1 = Z[0, 0]
        else:
            self_ = np.mean(np.diag(Z)); mut = np.mean(Z[~np.eye(n, dtype=bool)])
            z1 = self_ - mut
        out[k] = z1                              # ohm per kft
    return out


def base(name):
    return name.split(".")[0].lower()


def parse_123(folder, linecodes):
    z1 = parse_linecodes(linecodes)
    merge = {"150r": "149", "150": "149", "9r": "9", "25r": "25", "160r": "160"}
    m = lambda b: merge.get(b, b)
    edges = []
    for raw in open(os.path.join(folder, "IEEE123Master.dss")):
        ln = raw.split("!")[0].strip()
        if not ln.lower().startswith("new line."):
            continue
        name = ln.split()[1]
        b1 = m(base(re.search(r"Bus1=(\S+)", ln, re.I).group(1)))
        b2 = m(base(re.search(r"Bus2=(\S+)", ln, re.I).group(1)))
        if re.search(r"switch=yes", ln, re.I):
            if "_open" in b2 or name.lower() in ("line.sw7", "line.sw8", "line.sw6"):
                continue                         # open ties; Sw6 feeds unloaded XFM-1
            if b1 == b2:
                continue
            edges.append((b1, b2, 1e-3, 0.0, name)); continue
        code = re.search(r"LineCode=(\S+)", ln, re.I).group(1)
        L = float(re.search(r"Length=(\S+)", ln, re.I).group(1))
        z = z1[code] * L
        edges.append((b1, b2, z.real, z.imag, name))
    loads = {}
    for raw in open(os.path.join(folder, "IEEE123Loads.DSS")):
        ln = raw.split("!")[0].strip()
        if not ln.lower().startswith("new load."):
            continue
        b = m(base(re.search(r"Bus1=(\S+)", ln, re.I).group(1)))
        p = float(re.search(r"kW=(\S+)", ln, re.I).group(1))
        q = float(re.search(r"kvar=(\S+)", ln, re.I).group(1))
        P, Q = loads.get(b, (0.0, 0.0)); loads[b] = (P + p, Q + q)
    for raw in open(os.path.join(folder, "IEEE123Master.dss")):
        ln = raw.split("!")[0].strip()
        if ln.lower().startswith("new capacitor."):
            b = m(base(re.search(r"Bus1=(\S+)", ln, re.I).group(1)))
            q = float(re.search(r"kVAR=(\S+)", ln, re.I).group(1))
            P, Q = loads.get(b, (0.0, 0.0)); loads[b] = (P, Q - q)
    # orient edges away from the source (BFS) and number buses, source = 1
    adj = {}
    for a, b, r, x, nm in edges:
        adj.setdefault(a, []).append((b, r, x, nm)); adj.setdefault(b, []).append((a, r, x, nm))
    order, parent, seen = ["149"], {}, {"149"}
    i = 0
    while i < len(order):
        u = order[i]; i += 1
        for v, r, x, nm in sorted(adj[u], key=lambda t: t[0]):
            if v not in seen:
                seen.add(v); parent[v] = (u, r, x, nm); order.append(v)
    assert len(seen) == len(adj), "123-bus network not connected"
    assert len(edges) == len(adj) - 1, "123-bus network not radial"
    num = {b: k + 1 for k, b in enumerate(order)}
    branches = np.array([[num[parent[v][0]], num[v], parent[v][1], parent[v][2]] for v in order[1:]])
    L = np.array([[num[b], loads.get(b, (0, 0))[0], loads.get(b, (0, 0))[1]] for b in order])
    return branches, L, num


def main(case69, folder123, linecodes):
    br69, ld69 = parse_case69(case69)
    np.savetxt(os.path.join(HERE, "feeder69_branches.csv"), br69, delimiter=",",
               header="from,to,r_ohm,x_ohm", comments="", fmt=["%d", "%d", "%.6f", "%.6f"])
    np.savetxt(os.path.join(HERE, "feeder69_loads.csv"), ld69, delimiter=",",
               header="bus,p_kw,q_kvar", comments="", fmt=["%d", "%.3f", "%.3f"])
    br, ld, num = parse_123(folder123, linecodes)
    np.savetxt(os.path.join(HERE, "feeder123_branches.csv"), br, delimiter=",",
               header="from,to,r_ohm,x_ohm", comments="", fmt=["%d", "%d", "%.6f", "%.6f"])
    np.savetxt(os.path.join(HERE, "feeder123_loads.csv"), ld, delimiter=",",
               header="bus,p_kw,q_kvar", comments="", fmt=["%d", "%.3f", "%.3f"])
    with open(os.path.join(HERE, "feeder123_busmap.csv"), "w") as f:
        f.write("index,ieee_name\n")
        for b, k in sorted(num.items(), key=lambda t: t[1]):
            f.write(f"{k},{b}\n")
    print(f"69-bus: {len(ld69)} buses, {len(br69)} branches, load {ld69[:, 1].sum():.1f} kW / {ld69[:, 2].sum():.1f} kVAr")
    print(f"123-bus: {len(ld)} buses, {len(br)} branches, load {ld[:, 1].sum():.1f} kW / net Q {ld[:, 2].sum():.1f} kVAr")


if __name__ == "__main__":
    main(*sys.argv[1:4])
