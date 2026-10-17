"""PREREG14 A: train mlp_mid and cnnlstm_small under the frozen protocol (run_learned.run with two model kinds added).
Usage: python3 mid_run.py <ds> <kind>"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1"); os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
sys.path.insert(0, RUN); os.chdir(HERE)
import numpy as np
import frozen_benchmark as fb, fastfwd, nab_data as nd

fb.MACS["mlp_mid"] = 12 * 64 + 64 * 32 + 32          # 2,848
fb.MACS["cnnlstm_small"] = 12 * 3 * 8 + 12 * 4 * 8 * 16 + 8   # 6,440
SEQ = ("cnn_lstm", "cnnlstm_small")

_build = fb.build
def build(kind, seed):
    from tensorflow import keras
    from tensorflow.keras import layers
    if kind == "mlp_mid":
        keras.utils.set_random_seed(int(seed))
        m = keras.Sequential([layers.Input((fb.LOOK_BACK,)), layers.Dense(64, activation="relu"),
                              layers.Dense(32, activation="relu"), layers.Dense(1)])
        m.compile("adam", "mse"); return m
    if kind == "cnnlstm_small":
        keras.utils.set_random_seed(int(seed))
        m = keras.Sequential([layers.Input((fb.LOOK_BACK, 1)), layers.Conv1D(8, 3, activation="relu", padding="same"),
                              layers.LSTM(8), layers.Dense(1)])
        m.compile("adam", "mse"); return m
    return _build(kind, seed)
fb.build = build

def extract(m, kind):
    W = m.get_weights()
    if kind == "mlp_mid": return ("mlp_mid", *W)
    if kind in SEQ: return ("cnn_lstm", *W)
    return ("lightweight", *W)
_forward = fastfwd.forward
def forward(P, Z):
    if P[0] == "mlp_mid":
        _, k1, b1, k2, b2, k3, b3 = P
        h = np.maximum(Z @ k1 + b1, 0.0); h = np.maximum(h @ k2 + b2, 0.0)
        return (h @ k3 + b3).reshape(-1)
    return _forward(P, Z)
def verify(m, kind, rng=None, n=64, tol=2e-4):
    rng = rng or np.random.default_rng(0)
    Z = rng.standard_normal((n, m.input_shape[1])).astype("float32")
    zz = Z[..., None] if kind in SEQ else Z
    ref = np.asarray(m(zz, training=False)).reshape(-1).astype(np.float64)
    got = forward(extract(m, kind), Z.astype(np.float64))
    err = np.abs(ref - got).max(); scale = max(1e-8, np.abs(ref).max())
    assert err / scale <= tol, ("fast path disagrees with Keras", kind, err / scale)
    return err

def wrap_fast(m, kind, mx, sx, my, sy):
    verify(m, kind); P = extract(m, kind)
    return lambda W: forward(P, (np.asarray(W, dtype=np.float64) - mx) / sx) * sy + my

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
        if kind in SEQ: Xin = Xin[..., None]      # generalised: the only change from run_learned.run
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
    ds, kind = sys.argv[1], sys.argv[2]
    if ds == "intel_time": A, unit = np.load(os.path.join(HERE, "../lb24/intel_time.npy")), 1.0
    else:
        os.chdir(RUN); A, unit = nd.load(ds); os.chdir(HERE)
    run(A, ds, unit, [kind], range(20), f"learned_{ds}_{kind}.json")
