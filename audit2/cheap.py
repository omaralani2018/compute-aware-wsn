"""Independent recompute of the cheap arms (persistence / expsm / linear extrap)
for all 11 streams at tau in {0.05,0.10,0.20,0.50}. Selection on VALIDATION.
Writes $WORKDIR/audit2/cheap.json. Does not touch any existing file."""
import sys, json, numpy as np
import os; WORKDIR = os.path.abspath(os.environ.get("WORKDIR", "."))   # the directory the runs were made in (see README)
sys.path.insert(0, WORKDIR)
os.chdir(WORKDIR)
from dual_prediction_fix import dual_prediction_suppress as dps
import frozen_benchmark as fb
import nab_data

DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
      "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
TAUS = [0.05, 0.10, 0.20, 0.50]
LB = 12

def load(ds):
    if ds == "intel_time":
        return np.load("intel_time.npy"), 1.0
    return nab_data.load(ds)

def ev(X, f, tau, unit):
    return dps(X, f, tau=tau, look_back=LB, train_frac=1e-6, unit=unit)

out = {}
for ds in DS:
    A, unit = load(ds)
    A = np.asarray(A, float)
    if A.ndim == 1: A = A[:, None]
    T = len(A); tr = int(0.5 * T); va = int(0.7 * T)
    VAL = A[tr - LB:va]; TEST = A[va - LB:]
    TRAIN = A[:tr]
    d = np.diff(TRAIN, axis=0)
    fast = float(np.std(d) / np.std(TRAIN)) if A.shape[1] == 1 else None
    rec = {"T": T, "tr": tr, "va": va, "unit": unit, "N": A.shape[1],
           "fastness_train": fast, "n_test_steps": (len(TEST) - LB) * A.shape[1],
           "n_val_steps": (len(VAL) - LB) * A.shape[1], "taus": {}}
    for tau in TAUS:
        r = {}
        # open-loop: fraction of TRAINING-span first differences within tau (scaled by unit)
        r["openloop_train_frac"] = float((np.abs(d) / unit <= tau).mean())
        # also test-span open-loop fraction for reference
        dt = np.diff(A[va:], axis=0)
        r["openloop_test_frac"] = float((np.abs(dt) / unit <= tau).mean())
        pv = ev(VAL, None, tau, unit); pt = ev(TEST, None, tau, unit)
        r["persistence"] = {"val": pv["rho"], "test": pt["rho"],
                            "test_mean_err": pt["mean_err"], "test_max_err": pt["max_err"],
                            "n_tx_test": pt["n_tx"]}
        # expsm
        best = (-1.0, None); vals = {}
        for a in fb.ALPHA_GRID:
            rv = ev(VAL, fb.make_expsm(a), tau, unit)["rho"]
            vals[str(a)] = rv
            if rv > best[0]: best = (rv, a)
        et = ev(TEST, fb.make_expsm(best[1]), tau, unit)
        r["expsm"] = {"alpha": best[1], "val": best[0], "test": et["rho"],
                      "test_max_err": et["max_err"], "val_grid": vals}
        # linear extrap
        best = (-1.0, None); vals = {}
        for k in fb.LIN_K_GRID:
            rv = ev(VAL, fb.make_ar(k), tau, unit)["rho"]
            vals[str(k)] = rv
            if rv > best[0]: best = (rv, k)
        lt = ev(TEST, fb.make_ar(best[1]), tau, unit)
        r["linear_extrap"] = {"K": best[1], "val": best[0], "test": lt["rho"],
                              "test_max_err": lt["max_err"], "val_grid": vals}
        # reference: highest VALIDATION suppression; ties -> persistence, expsm, linear
        arms = [("persistence", r["persistence"]["val"], r["persistence"]["test"], 0),
                ("expsm", r["expsm"]["val"], r["expsm"]["test"], 12),
                ("linear_extrap", r["linear_extrap"]["val"], r["linear_extrap"]["test"],
                 r["linear_extrap"]["K"])]
        bestarm = arms[0]
        for arm in arms[1:]:
            if arm[1] > bestarm[1]: bestarm = arm
        r["reference"] = {"arm": bestarm[0], "val": bestarm[1], "test": bestarm[2], "mac": bestarm[3]}
        # check for near-ties (within 1e-12) that the tie rule decides
        r["val_ties"] = [(a[0], b[0]) for i, a in enumerate(arms) for b in arms[i+1:]
                         if abs(a[1] - b[1]) < 1e-12]
        rec["taus"][str(tau)] = r
        print(f"{ds:18s} tau={tau:<5} pers val={pv['rho']:.4f} test={pt['rho']:.4f} | "
              f"expsm a={r['expsm']['alpha']} val={r['expsm']['val']:.4f} test={r['expsm']['test']:.4f} | "
              f"lin K={r['linear_extrap']['K']} val={r['linear_extrap']['val']:.4f} test={r['linear_extrap']['test']:.4f} | "
              f"ref={bestarm[0]} {bestarm[2]:.4f} ties={r['val_ties']}", flush=True)
    out[ds] = rec
json.dump(out, open(f"{WORKDIR}/audit2/cheap.json", "w"), indent=1)
print("saved")
