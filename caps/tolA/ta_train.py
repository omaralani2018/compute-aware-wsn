"""Tolerance-aware training of the lightweight predictor (PREREG3.txt).
Identical to run_learned.run except the loss and the joint (c, epoch) selection."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
sys.path.insert(0, RUN)
import numpy as np
import frozen_benchmark as fb, nab_data as nd
from run_learned import wrap_fast
import tensorflow as tf

C_GRID = [0.25, 0.5, 1.0]

def eps_loss(eps):
    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, y_pred.dtype), tf.shape(y_pred))
        return tf.reduce_mean(tf.maximum(tf.abs(y_pred - y_true) - eps, 0.0))
    return loss

def run(ds, seeds, out):
    if ds == "intel_time": A, unit = np.load(os.path.join(RUN, "intel_time.npy")), 1.0
    else: A, unit = nd.load(ds)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    VAL, TEST = A[tr - fb.LOOK_BACK:va], A[va - fb.LOOK_BACK:]
    Xtr, Ytr = fb.windows(A, A, fb.LOOK_BACK, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    Xin, Yin = (Xtr - mx) / sx, (Ytr - my) / sy
    rows = json.load(open(out))["entries"] if os.path.exists(out) else []
    done = {r["seed"] for r in rows}
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    for sd in seeds:
        if sd in done: continue
        t0 = time.time(); best = (-1, None, None, None); per_c = {}
        for c in C_GRID:
            eps = c * fb.TAU * unit / sy
            m = fb.build("lightweight", sd)                     # same init as the MSE run
            m.compile("adam", eps_loss(eps))
            bc = -1
            for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP + 1, fb.CKPT_EVERY):
                m.fit(Xin, Yin, epochs=fb.CKPT_EVERY, batch_size=64, verbose=0, validation_split=0.15)
                f = wrap_fast(m, "lightweight", mx, sx, my, sy)
                r = fb.dps(VAL, f, tau=fb.TAU, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)["rho"]
                bc = max(bc, r)
                if r > best[0]: best = (r, c, ep, [w.copy() for w in m.get_weights()])
            per_c[str(c)] = bc
        m = fb.build("lightweight", sd); m.set_weights(best[3])
        f = wrap_fast(m, "lightweight", mx, sx, my, sy)
        t_ = fb.dps(TEST, f, tau=fb.TAU, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)
        vr = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=fb.TAU, look_back=fb.LOOK_BACK,
                       train_frac=1e-6, unit=unit, strict=True)
        rows.append({"seed": int(sd), "c": best[1], "selected_epoch": best[2], "val_rho": best[0],
                     "val_rho_per_c": per_c, "test_rho": t_["rho"], "test_max_err": t_["max_err"],
                     "receiver_verified": vr["bound_holds"], "secs": round(time.time() - t0, 1)})
        tmp = out + ".tmp"
        json.dump({"dataset": ds, "unit": unit, "entries": rows}, open(tmp, "w"), indent=1); os.replace(tmp, out)
        np.savez(os.path.join(HERE, "weights", f"{ds}_ta_s{sd}.npz"), *m.get_weights())
        print("  %s seed %2d c=%.2f ep%3d val=%.3f test=%.3f %.0fs" % (ds, sd, best[1], best[2], best[0], t_["rho"], time.time() - t0), flush=True)

if __name__ == "__main__":
    ds, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    run(ds, range(s0, s1), os.path.join(HERE, f"ta_{ds}.json"))
