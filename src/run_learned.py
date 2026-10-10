"""Learned arms of the frozen benchmark, using the verified NumPy forward pass.

The protocol is unchanged: same splits, tolerance, epoch cap, checkpoint
interval, seed set and selection rule as frozen_benchmark.py. Only the
implementation of the forward pass differs, and it is asserted equivalent to
Keras (<=2e-4 relative) for EVERY checkpoint before that checkpoint is scored.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS","1"); os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")
import sys, json, time, numpy as np
import frozen_benchmark as fb, fastfwd, nab_data as nd

def wrap_fast(m, kind, mx, sx, my, sy):
    fastfwd.verify_against_keras(m, kind)      # per-checkpoint equivalence check
    P = fastfwd.extract(m, kind)
    def f(W):
        z = (np.asarray(W, dtype=np.float64) - mx) / sx
        return fastfwd.forward(P, z) * sy + my
    return f

def run(A, name, unit, kinds, seeds, out):
    A = np.asarray(A, float)
    if A.ndim == 1: A = A[:, None]
    T = A.shape[0]; tr, va = int(fb.SPLIT[0]*T), int(fb.SPLIT[1]*T)
    VAL, TEST = A[tr-fb.LOOK_BACK:va], A[va-fb.LOOK_BACK:]
    assert tr < va < T
    rng = np.random.default_rng(0)
    Xtr, Ytr = fb.windows(A, A, fb.LOOK_BACK, tr, rng)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    rows = json.load(open(out))["entries"] if os.path.exists(out) else []
    done = {(r["predictor"], r["seed"]) for r in rows}
    for kind in kinds:
        Xin = (Xtr - mx)/sx
        if kind == "cnn_lstm": Xin = Xin[..., None]
        Yin = (Ytr - my)/sy
        for sd in seeds:
            if (kind, sd) in done: continue
            t0 = time.time(); m = fb.build(kind, sd); best = (-1, None, None)
            for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP+1, fb.CKPT_EVERY):
                m.fit(Xin, Yin, epochs=fb.CKPT_EVERY, batch_size=64,
                      verbose=0, validation_split=0.15)
                f = wrap_fast(m, kind, mx, sx, my, sy)
                r = fb.dps(VAL, f, tau=fb.TAU, look_back=fb.LOOK_BACK,
                           train_frac=1e-6, unit=unit)["rho"]
                if r > best[0]: best = (r, ep, [w.copy() for w in m.get_weights()])
            m.set_weights(best[2]); f = wrap_fast(m, kind, mx, sx, my, sy)
            t_ = fb.dps(TEST, f, tau=fb.TAU, look_back=fb.LOOK_BACK,
                        train_frac=1e-6, unit=unit)
            vr = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=fb.TAU,
                           look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit,
                           strict=True)
            e = fb.energy_report(kind, t_["rho"], t_["n_steps"], t_["n_init_reports"])
            rows.append({"predictor": kind, "seed": int(sd), "selected_epoch": best[1],
                         "val_rho": best[0], "test_rho": t_["rho"],
                         "test_mean_err": t_["mean_err"], "test_max_err": t_["max_err"],
                         "test_rmse": t_["rmse"], "test_nonfinite": t_["n_nonfinite_forecasts"],
                         "receiver_verified": vr["bound_holds"], "energy": e,
                         "secs": round(time.time()-t0, 1)})
            tmp = out + ".tmp"                      # atomic: a kill cannot corrupt
            json.dump({"dataset": name, "unit": unit, "T": T, "split": [tr, va],
                       "tau": fb.TAU, "entries": rows}, open(tmp, "w"), indent=1)
            os.replace(tmp, out)
            os.makedirs("weights", exist_ok=True)
            np.savez(f"weights/{name}_{kind}_s{sd}.npz", *m.get_weights())
            print("  %-10s seed %2d  ep%3d  val=%.3f test=%.3f  %.0fs"
                  % (kind, sd, best[1], best[0], t_["rho"], time.time()-t0), flush=True)

if __name__ == "__main__":
    ds, kinds, s0, s1 = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
    if ds == "intel":
        A, unit = fb.load_intel(fb.INTEL_SPEC), 1.0
    elif ds == "intel_time":
        A, unit = np.load("intel_time.npy"), 1.0
    else:
        A, unit = nd.load(ds)
    print(f"[{ds}] T={len(A)} unit={unit:.4g} kinds={kinds} seeds {s0}..{s1-1}", flush=True)
    run(A, ds, unit, kinds, range(s0, s1), f"learned_{ds}.json")
