"""Tolerance sweep for the lightweight arm. Same frozen protocol, except the
checkpoint is re-selected on validation separately at each tolerance, exactly
as the cheap arms re-select alpha and K at each tolerance."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS","1"); os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")
import sys, json, time, numpy as np
import frozen_benchmark as fb, fastfwd, nab_data as nd
from run_learned import wrap_fast
TAUS = [0.05, 0.10, 0.20, 0.50]
def main(ds):
    A, unit = (fb.load_intel(fb.INTEL_SPEC), 1.0) if ds == "intel" else (np.load("intel_time.npy"), 1.0) if ds == "intel_time" else nd.load(ds)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = len(A); tr, va = int(fb.SPLIT[0]*T), int(fb.SPLIT[1]*T)
    VAL, TEST = A[tr-fb.LOOK_BACK:va], A[va-fb.LOOK_BACK:]
    rng = np.random.default_rng(0)
    Xtr, Ytr = fb.windows(A, A, fb.LOOK_BACK, tr, rng)
    mx, sx = Xtr.mean(0), Xtr.std(0)+1e-8; my, sy = float(Ytr.mean()), float(Ytr.std()+1e-8)
    out = f"tau_{ds}.json"
    rows = json.load(open(out)) if os.path.exists(out) else []
    done = {r["seed"] for r in rows}
    for sd in range(20):
        if sd in done: continue
        t0 = time.time(); m = fb.build("lightweight", sd); ck = []
        for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP+1, fb.CKPT_EVERY):
            m.fit((Xtr-mx)/sx, (Ytr-my)/sy, epochs=fb.CKPT_EVERY, batch_size=64,
                  verbose=0, validation_split=0.15)
            f = wrap_fast(m, "lightweight", mx, sx, my, sy)
            ck.append((ep, [w.copy() for w in m.get_weights()],
                       {t: fb.dps(VAL, f, tau=t, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)["rho"] for t in TAUS}))
        rec = {"seed": sd}
        for t in TAUS:
            best = ck[0]
            for c in ck[1:]:
                if c[2][t] > best[2][t]: best = c        # ties keep the earlier epoch
            m.set_weights(best[1]); f = wrap_fast(m, "lightweight", mx, sx, my, sy)
            r = fb.dps(TEST, f, tau=t, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)
            v = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=t, look_back=fb.LOOK_BACK,
                          train_frac=1e-6, unit=unit, strict=True)
            rec[str(t)] = {"epoch": best[0], "val": best[2][t], "test": r["rho"],
                           "max_err": r["max_err"], "verified": v["bound_holds"]}
        rows.append(rec)
        json.dump(rows, open(out+".tmp","w"), indent=1); os.replace(out+".tmp", out)
        print(ds, sd, "%.0fs" % (time.time()-t0), flush=True)
if __name__ == "__main__":
    main(sys.argv[1])
