"""Closed-loop tolerance-aware training (PREREG5.txt)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
TA = os.path.abspath(os.path.join(HERE, "../tolA"))
sys.path.insert(0, RUN); sys.path.insert(0, TA)
import numpy as np
import frozen_benchmark as fb, nab_data as nd
from run_learned import wrap_fast
from ta_train import eps_loss

LB, TAU = fb.LOOK_BACK, fb.TAU
ROUNDS, EP_ROUND, CK = 3, 50, 25

def recon_windows(X, f, unit, rng, n=fb.MAX_WINDOWS):
    """Closed-loop run over X; return (inputs from reconstructed series, true targets)."""
    R = X.copy()
    for t in range(LB, X.shape[0]):
        p = np.asarray(f(R[t - LB:t].T), float).reshape(-1)
        sup = (np.abs(X[t] - p) / unit) <= TAU
        R[t] = np.where(sup, p, X[t])
    Xs, Ys = [], []
    for j in range(X.shape[1]):
        for t in range(LB, X.shape[0]):
            Xs.append(R[t - LB:t, j]); Ys.append(X[t, j])
    Xs, Ys = np.array(Xs), np.array(Ys)
    if len(Xs) > n:
        i = rng.choice(len(Xs), n, replace=False); Xs, Ys = Xs[i], Ys[i]
    return Xs, Ys

def run(ds, seeds, out):
    if ds == "intel_time": A, unit = np.load(os.path.join(RUN, "intel_time.npy")), 1.0
    else:
        cwd = os.getcwd(); os.chdir(RUN); A, unit = nd.load(ds); os.chdir(cwd)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    TRS, VAL, TEST = A[:tr], A[tr - LB:va], A[va - LB:]
    Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    ta = {r["seed"]: r for r in json.load(open(os.path.join(TA, f"ta_{ds}.json")))["entries"]}
    rows = json.load(open(out))["entries"] if os.path.exists(out) else []
    done = {r["seed"] for r in rows}
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    for sd in seeds:
        if sd in done: continue
        t0 = time.time(); c = ta[sd]["c"]; eps = c * TAU * unit / sy
        w = np.load(os.path.join(TA, "weights", f"{ds}_ta_s{sd}.npz")); W0 = [w["arr_%d" % i] for i in range(len(w.files))]
        m = fb.build("lightweight", sd); m.set_weights(W0); m.compile("adam", eps_loss(eps))
        f = wrap_fast(m, "lightweight", mx, sx, my, sy)
        v0 = fb.dps(VAL, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=unit)["rho"]
        assert abs(v0 - ta[sd]["val_rho"]) < 1e-9, (ds, sd, v0, ta[sd]["val_rho"])
        best = (v0, 0, 0, [x.copy() for x in W0]); trace = [(0, 0, v0)]
        DX, DY = [Xtr], [Ytr]; rng = np.random.default_rng(1000 + sd)
        for r in range(1, ROUNDS + 1):
            rx, ry = recon_windows(TRS, f, unit, rng); DX.append(rx); DY.append(ry)
            Xin = (np.concatenate(DX) - mx) / sx; Yin = (np.concatenate(DY) - my) / sy
            for ep in range(CK, EP_ROUND + 1, CK):
                m.fit(Xin, Yin, epochs=CK, batch_size=64, verbose=0)
                f = wrap_fast(m, "lightweight", mx, sx, my, sy)
                v = fb.dps(VAL, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=unit)["rho"]
                trace.append((r, ep, v))
                if v > best[0]: best = (v, r, ep, [x.copy() for x in m.get_weights()])
        m.set_weights(best[3]); f = wrap_fast(m, "lightweight", mx, sx, my, sy)
        t_ = fb.dps(TEST, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=unit)
        vr = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=TAU, look_back=LB, train_frac=1e-6, unit=unit, strict=True)
        rows.append({"seed": int(sd), "c": c, "round": best[1], "epoch": best[2], "val_rho": best[0],
                     "test_rho": t_["rho"], "receiver_verified": vr["bound_holds"], "trace": trace,
                     "secs": round(time.time() - t0, 1)})
        tmp = out + ".tmp"; json.dump({"dataset": ds, "unit": unit, "entries": rows}, open(tmp, "w"), indent=1); os.replace(tmp, out)
        np.savez(os.path.join(HERE, "weights", f"{ds}_clt_s{sd}.npz"), *m.get_weights())
        print("  %s seed %2d round %d ep%2d val=%.3f (ta %.3f) test=%.3f (ta %.3f) %.0fs" % (
            ds, sd, best[1], best[2], best[0], v0, t_["rho"], ta[sd]["test_rho"], time.time() - t0), flush=True)

if __name__ == "__main__":
    ds, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    run(ds, range(s0, s1), os.path.join(HERE, f"clt_{ds}.json"))
