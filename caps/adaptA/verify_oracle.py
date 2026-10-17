"""Independent check of oracle_switch.py results.
1) Static rho via the frozen evaluator dual_prediction_suppress (a different code path).
2) Energy arithmetic of E_R, E_L, G from the stored rows.
3) A scalar, per-node re-implementation of the block oracle for spot seeds.
"""
import os, sys, json, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import oracle_switch as O
fb = O.fb; ER, EOP, LB, TAU = O.ER, O.EOP, O.LB, O.TAU

def scalar_oracle(x, fR, fL, mR, mL, B, unit):
    T = len(x); r = x.astype(float).copy(); cur = -1; E = 0.0
    def fc(f, w): return w[-1] if f is None else float(np.asarray(f(w[None, :])).reshape(-1)[0])
    for a in range(LB, T, B):
        b = min(a + B, T); best = None
        for k, (f, m) in enumerate(((fR, mR), (fL, mL))):
            rr = r.copy(); tx = 0
            for t in range(a, b):
                p = fc(f, rr[t - LB:t])
                if abs(x[t] - p) / unit <= TAU: rr[t] = p
                else: rr[t] = x[t]; tx += 1
            e = tx * ER + (b - a) * m * EOP + (ER if (cur >= 0 and cur != k) else 0.0)
            if best is None or e < best[0]: best = (e, k, rr)
        E += best[0]; cur = best[1]; r = best[2]
    return E

out = []; bad = 0
for f in sorted(glob.glob(os.path.join(HERE, "oracle_*_lightweight.json"))):
    d = json.load(open(f)); ds = d["ds"]; n = d["n_readings"]
    A, unit = O.load(ds); A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; va = int(fb.SPLIT[1] * T); X = A[va - LB:]
    _, fR, mR, _, _ = O.ref_arm(ds)
    rR = fb.dps(X, fR, tau=TAU, look_back=LB, train_frac=1e-6, unit=unit)["rho"]
    for row in d["rows"]:
        ok = abs(row["rhoR"] - rR) < 1e-12
        ok &= abs(row["E_R"] - ((1 - row["rhoR"]) * n * ER + n * mR * EOP)) < 1e-9
        ok &= abs(row["E_L"] - ((1 - row["rhoL"]) * n * ER + n * 208 * EOP)) < 1e-9
        for B in (96, 48, 192):
            g = 1 - row["B%d" % B]["E"] / min(row["E_R"], row["E_L"])
            ok &= abs(g - row["B%d" % B]["G_best"]) < 1e-12
        bad += (not ok)
    # spot check: seed 0, B=96, scalar re-implementation (intel_time: first 3 nodes only)
    r0 = d["rows"][0]
    fLd = O.learned_arm(A, "lightweight", os.path.normpath(os.path.join(HERE, r0["weights"])))
    if X.shape[1] == 1:
        Es = scalar_oracle(X[:, 0], fR, fLd, mR, 208, 96, unit)
        Ev = O.oracle(X, [fR, fLd], [mR, 208], 96, unit)[0]
    else:
        Es = sum(scalar_oracle(X[:, j], fR, fLd, mR, 208, 96, unit) for j in range(3))
        Ev = O.oracle(X[:, :3], [fR, fLd], [mR, 208], 96, unit)[0]
    agree = abs(Es - Ev) <= 1e-9 * max(1, abs(Ev))
    bad += (not agree)
    out.append((ds, len(d["rows"]), agree, Es, Ev))
    print("%-18s rows=%d arithmetic+static ok, scalar oracle agrees=%s (%.6f vs %.6f)" % (ds, len(d["rows"]), agree, Es, Ev), flush=True)
print("FAILURES:", bad)
