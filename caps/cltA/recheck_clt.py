"""Version 1.6. Like tolA/recheck.py for the CLT models: re-evaluates nine saved models (three streams x seeds 0, 7, 19)
in closed loop with oracle_switch.static and checks that test suppression equals clt_<stream>.json (prints the
number of mismatches). Run in caps/cltA after setup.sh."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.abspath("../adaptA")); import oracle_switch as O
fb = O.fb; bad = 0
for ds in ["twitter_aapl", "occupancy", "taxi"]:
    A, unit = O.load(ds); A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; va = int(fb.SPLIT[1] * T); X = A[va - O.LB:]
    e = {r["seed"]: r for r in json.load(open(f"clt_{ds}.json"))["entries"]}
    for sd in (0, 7, 19):
        f = O.learned_arm(A, "lightweight", f"weights/{ds}_clt_s{sd}.npz")
        n = X.shape[0] - O.LB; rho = 1 - O.static(X, f, unit).sum() / n
        ok = abs(rho - e[sd]["test_rho"]) < 1e-9; bad += not ok
        print(ds, sd, round(rho, 6), round(e[sd]["test_rho"], 6), "ok" if ok else "MISMATCH")
print("mismatches", bad)
