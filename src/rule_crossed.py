"""
Version 1.4. Crossed rule comparison with a common boundary convention, a
retrospective diagnostic added in version 1.4 (Section 6.8). Inside run/, after
matched_baselines.py has written matched_<stream>.json:
    python3 rule_crossed.py <stream>      -> rule_crossed_<stream>.json

For each scheme and stream, the two parameter sets already selected on validation
(under the released rule and under the single-error rule) are each scored on the
test span under:
  cons_lt : suppress iff the last W errors are all  < tau   (released code, faithful)
  cons_le : suppress iff the last W errors are all <= tau   (same boundary as Sec. 3.3)
  single_le: suppress iff the current error        <= tau   (Section 3.3, as published)
  single_lt: suppress iff the current error         < tau
The controlled comparison is cons_le vs single_le (only the consecutive requirement
differs). Data, tau, warm-up, initial state identical; scheme state evolves under its
own decisions; independent receiver checked. Reports rho, transmissions, errors.
"""
import json, sys, numpy as np, pandas as pd
import matched_baselines as mb
from nab_data import NAB
L, TAU = mb.LOOK_BACK, mb.TAU


def load(name):
    if name == "intel_time": return np.load("intel_time.npy"), 1.0
    v = pd.read_csv(NAB[name])["value"].to_numpy(float)[:, None]
    return v, float(v[:int(mb.SPLIT[0] * len(v))].std())


def maker(scheme, p, n):
    if scheme == "LMS": return (lambda w: mb.LMS(n, p["N"], p["mu"], w)), p["N"]
    return (lambda w: mb.AMDR(n, p["w_f"], p["w_s"], p["alpha"], w)), p["w_f"]


def run(X, make, rule, W, unit):
    T, n = X.shape; s = make(X[:L]); eh = np.full((n, W), np.inf)
    packets = np.zeros((T, n), bool); recon = X.copy(); ties = 0
    for t in range(L, T):
        pred = s.predict(); err = np.abs(X[t] - pred) / unit
        ties += int((err == TAU).sum())
        if rule == "single_le": sup = err <= TAU
        elif rule == "single_lt": sup = err < TAU
        else:
            eh = np.concatenate([eh[:, 1:], err[:, None]], 1)
            sup = (eh < TAU).all(1) if rule == "cons_lt" else (eh <= TAU).all(1)
        sent = ~sup; value = np.where(sent, X[t], pred)
        s.update(sent, value, pred); recon[t] = value; packets[t] = sent
    rec = mb.receiver(X, make, packets, TAU, unit)
    assert np.array_equal(rec, recon)
    e = np.abs(rec[L:] - X[L:])
    assert e.max() <= TAU * unit + 1e-12
    ntx = int(packets[L:].sum()); nst = (T - L) * n
    return {"rho": 1 - ntx / nst, "n_tx": ntx, "n_steps": nst, "ties_at_tau": ties,
            "mean_abs_err": float(e.mean()), "max_abs_err": float(e.max()),
            "max_err_over_tau_unit": float(e.max() / (TAU * unit))}


name = sys.argv[1]
A, unit = load(name); n = A.shape[1]; T = A.shape[0]
X = A[int(mb.SPLIT[1] * T) - L:]
pub = json.load(open(f"matched_{name}.json"))
runs = {(r["scheme"], r["rule"]): r for r in pub["runs"]}
out = {"dataset": name, "unit": unit, "cells": {}}
for sc in ("LMS", "AMDR"):
    for src in ("authors", "ours"):
        r = runs[(sc, src)]; p = r["selected"]; mk, W = maker(sc, p, n)
        cell = {"params": p, "macs": r["macs"], "val_all_zero": all(v == 0 for _, v in r["val_grid"])}
        for rule in ("cons_lt", "cons_le", "single_le", "single_lt"):
            cell[rule] = run(X, mk, rule, W, unit)
        # existing published results reproduced
        pubrule = "cons_lt" if src == "authors" else "single_le"
        assert abs(cell[pubrule]["rho"] - r["test"]["rho"]) < 1e-12
        out["cells"][f"{sc}_sel_{src}"] = cell
        c = cell
        print(f"[{name}] {sc:4s} sel_{src:7s} consLT={c['cons_lt']['rho']:.4f} consLE={c['cons_le']['rho']:.4f} "
              f"singLE={c['single_le']['rho']:.4f} singLT={c['single_lt']['rho']:.4f} "
              f"ties={[c[k]['ties_at_tau'] for k in ('cons_lt','cons_le','single_le','single_lt')]}", flush=True)
json.dump(out, open(f"rule_crossed_{name}.json", "w"), indent=1)
