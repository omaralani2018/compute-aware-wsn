"""Option A, Step 1: block-oracle bound for sink-assisted switching (see PREREG.txt).

Exact closed loop with a shared reconstructed state, same send-on-delta rule as
dual_prediction_fix.dual_prediction_suppress. Static runs reuse the same simulator
and are checked against the stored test rho before any switching number is used.
"""
import os, sys, json, glob
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
sys.path.insert(0, RUN)
import numpy as np
import frozen_benchmark as fb, fastfwd, nab_data as nd

ER, EOP, LB, TAU = fb.E_R, fb.E_OP, fb.LOOK_BACK, fb.TAU
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]


def load(ds):
    if ds == "intel_time":
        return np.load(os.path.join(RUN, "intel_time.npy")), 1.0
    return nd.load(ds)


def ref_arm(ds):
    bt = MS[ds]["by_tau"]["0.1"]; sel = bt["selected"]
    if sel == "persistence": f = None
    elif sel == "expsm": f = fb.make_expsm(bt["alpha"])
    elif sel == "linear_extrap": f = fb.make_ar(bt["K"])
    else: raise ValueError(sel)
    return sel, f, bt["selected_macs"], bt["val"][sel], bt["test"][sel]["rho"]


def learned_arm(A, kind, wfile):
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr = int(fb.SPLIT[0] * T)
    Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    w = np.load(wfile); W = [w["arr_%d" % i] for i in range(len(w.files))]
    P = (kind, *W)
    def f(Win):
        z = (np.asarray(Win, dtype=np.float64) - mx) / sx
        return fastfwd.forward(P, z) * sy + my
    return f


def predict(f, R, t):
    if f is None: return R[t - 1].copy()
    return np.asarray(f(R[t - LB:t].T), float).reshape(-1)


def simulate(X, R, f, a, b, unit):
    """Advance shared state R in place over [a,b) with arm f. Returns tx count per node."""
    tx = np.zeros(X.shape[1], int)
    for t in range(a, b):
        p = predict(f, R, t)
        sup = (np.abs(X[t] - p) / unit) <= TAU          # inf/nan forecast -> transmit
        R[t] = np.where(sup, p, X[t]); tx += ~sup
    return tx


def static(X, f, unit):
    R = X.copy(); tx = simulate(X, R, f, LB, X.shape[0], unit)
    return tx


def oracle(X, arms, macs, B, unit):
    """Greedy block oracle. arms: [fR, fL]; per node choice; first block free choice."""
    T, N = X.shape; R = X.copy(); cur = np.full(N, -1)
    E = np.zeros(N); tx_tot = np.zeros(N, int); nsw = np.zeros(N, int); usedL = 0
    for a in range(LB, T, B):
        b = min(a + B, T); lo = a - LB
        trial = []
        for k, f in enumerate(arms):
            Rk = R[lo:b].copy()
            txk = simulate(X[lo:b], Rk, f, LB, b - lo, unit)
            ek = txk * ER + (b - a) * macs[k] * EOP + ((cur >= 0) & (cur != k)) * ER
            trial.append((ek, txk, Rk))
        ch = np.where(trial[1][0] < trial[0][0], 1, 0)
        for j in range(N):
            k = ch[j]; ek, txk, Rk = trial[k]
            R[lo:b, j] = Rk[:, j]; E[j] += ek[j]; tx_tot[j] += txk[j]
            if cur[j] >= 0 and cur[j] != k: nsw[j] += 1
            cur[j] = k
        usedL += int(ch.sum()) * (b - a)
    return E.sum(), tx_tot.sum(), nsw.sum(), usedL / ((T - LB) * N)


def run(ds, kind, Bs=(96, 48, 192), seeds=range(20), out=None):
    A, unit = load(ds); A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; va = int(fb.SPLIT[1] * T)
    X = A[va - LB:]; n = (X.shape[0] - LB) * X.shape[1]
    rname, fR, mR, rvalR, rtestR = ref_arm(ds)
    txR = static(X, fR, unit).sum(); rhoR = 1 - txR / n
    assert abs(rhoR - rtestR) < 1e-9, (ds, rhoR, rtestR)
    ER_static = txR * ER + n * mR * EOP
    mL = fb.MACS[kind]
    stored = {r["seed"]: r for r in json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"]
              if r["predictor"] == kind}
    wdirs = [os.path.join(HERE, "weights"), os.path.join(RUN, "weights")]
    rows = []
    for sd in seeds:
        wf = next((p for d in wdirs for p in [os.path.join(d, f"{ds}_{kind}_s{sd}.npz")] if os.path.exists(p)), None)
        if wf is None: continue
        fL = learned_arm(A, kind, wf)
        txL = static(X, fL, unit).sum(); rhoL = 1 - txL / n
        ok = abs(rhoL - stored[sd]["test_rho"]) <= 0.005
        EL = txL * ER + n * mL * EOP
        # compute-aware criterion on VALIDATION (per seed): pick L iff it pays there
        pickL = (stored[sd]["val_rho"] - rvalR) * ER > (mL - mR) * EOP
        Ecrit = EL if pickL else ER_static
        row = {"seed": sd, "weights": os.path.relpath(wf, HERE), "rhoR": rhoR, "rhoL": rhoL,
               "rhoL_stored": stored[sd]["test_rho"], "reproduced": bool(ok),
               "E_R": ER_static, "E_L": EL, "crit_picks_L": bool(pickL), "E_crit": Ecrit}
        for B in Bs:
            Es, txs, nsw, fracL = oracle(X, [fR, fL], [mR, mL], B, unit)
            row[f"B{B}"] = {"E": Es, "rho": 1 - txs / n, "switches": int(nsw), "fracL": fracL,
                            "G_best": 1 - Es / min(ER_static, EL), "G_crit": 1 - Es / Ecrit}
        rows.append(row)
        print(ds, kind, sd, "rhoR=%.3f rhoL=%.3f(%s) G96=%.3f sw=%d fracL=%.2f" % (
            rhoR, rhoL, "ok" if ok else "MISMATCH", row["B96"]["G_best"], row["B96"]["switches"],
            row["B96"]["fracL"]), flush=True)
    res = {"ds": ds, "kind": kind, "ref": rname, "ref_macs": mR, "n_readings": n, "rows": rows}
    json.dump(res, open(out or os.path.join(HERE, f"oracle_{ds}_{kind}.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    ds, kind = sys.argv[1], sys.argv[2]
    seeds = range(int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else range(20)
    out = sys.argv[5] if len(sys.argv) > 5 else None
    run(ds, kind, seeds=seeds, out=out)
