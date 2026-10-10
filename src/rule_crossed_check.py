"""Version 1.4. Independent scalar re-implementation (per node, plain Python loops,
no import of matched_baselines or rule_crossed) of LMS and AM-DR under the four rule
variants of rule_crossed.py. Re-scores the four cells (two schemes x two parameter sets) of each rule_crossed_<stream>.json, 44 in all,
and asserts that every transmission count is identical. Sums are taken left to right (lsum), as the
built-in sum() does up to Python 3.11; from Python 3.12 sum() uses compensated float summation, which
changed four Intel counts when the earlier version of this checker was run under Python 3.12. Inside run/:
    python3 rule_crossed_check.py <stream>"""
import json, sys, math, numpy as np, pandas as pd
from nab_data import NAB
L, TAU = 12, 0.10

def lsum(values):
    """Plain left-to-right float summation (independent of the Python version)."""
    s = 0.0
    for v in values: s += v
    return s

def load(name):
    if name == "intel_time": return np.load("intel_time.npy"), 1.0
    v = pd.read_csv(NAB[name])["value"].to_numpy(float)[:, None]
    return v, float(v[:int(0.5 * len(v))].std())

def lms_node(x, N, mu, rule, unit):
    hist = list(x[:N]); w = [0.0] * N
    for k in range(N, L):
        e = x[k] - lsum(a * b for a, b in zip(w, hist))
        w = [wi + mu * h * e for wi, h in zip(w, hist)]; hist = hist[1:] + [x[k]]
    eh = [math.inf] * N; ntx = 0
    for t in range(L, len(x)):
        pred = lsum(a * b for a, b in zip(w, hist)); err = abs(x[t] - pred) / unit
        if rule == "single_le": sup = err <= TAU
        elif rule == "single_lt": sup = err < TAU
        else:
            eh = eh[1:] + [err]
            sup = all(v < TAU for v in eh) if rule == "cons_lt" else all(v <= TAU for v in eh)
        if sup: val = pred
        else:
            ntx += 1; val = x[t]; e = x[t] - pred
            w = [wi + mu * h * e for wi, h in zip(w, hist)]
        hist = hist[1:] + [val]
    return ntx

def amdr_node(x, wf, ws, a, rule, unit):
    lam = 0.0; buf = list(x[L - (wf - 1):L])
    ssum = lsum(x[:ws]); scnt = float(ws)
    for k in range(ws, L):
        maf = lsum(x[k - wf + 1:k]) / (wf - 1); mas = ssum / scnt
        y = lam * maf + (1 - lam) * mas; lam = lam + a * (x[k] - y) * (maf - mas)
        ssum = x[k]; scnt = 1.0
    eh = [math.inf] * wf; ntx = 0
    for t in range(L, len(x)):
        maf = lsum(buf) / len(buf); mas = ssum / scnt
        pred = lam * maf + (1 - lam) * mas; err = abs(x[t] - pred) / unit
        if rule == "single_le": sup = err <= TAU
        elif rule == "single_lt": sup = err < TAU
        else:
            eh = eh[1:] + [err]
            sup = all(v < TAU for v in eh) if rule == "cons_lt" else all(v <= TAU for v in eh)
        if sup: val = pred; ssum += pred; scnt += 1
        else:
            ntx += 1; val = x[t]; lam = lam + a * (x[t] - pred) * (maf - mas); ssum = x[t]; scnt = 1.0
        buf = buf[1:] + [val]
    return ntx

name = sys.argv[1]
A, unit = load(name); T = A.shape[0]; va = int(0.70 * T); X = A[va - L:]
R = json.load(open(f"rule_crossed_{name}.json"))
n_cells = 0
for key, c in R["cells"].items():
    sc = key.split("_")[0]; p = c["params"]
    for rule in ("cons_lt", "cons_le", "single_le", "single_lt"):
        ntx = 0
        for j in range(X.shape[1]):
            x = [float(v) for v in X[:, j]]
            ntx += lms_node(x, p["N"], p["mu"], rule, unit) if sc == "LMS" else amdr_node(x, p["w_f"], p["w_s"], p["alpha"], rule, unit)
        assert ntx == c[rule]["n_tx"], (name, key, rule, ntx, c[rule]["n_tx"])
        n_cells += 1
print(f"{name}: {n_cells} runs, transmission counts identical")
