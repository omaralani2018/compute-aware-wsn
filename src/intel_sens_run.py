"""Intel preprocessing sensitivity: the full protocol of the main study at tau = 0.10 degC on one
preparation variant (intel_variants.py), evaluated on independent per-mote segments (segeval.py).

Unchanged from the main study: the five predictors and their MAC counts, the alpha and K grids,
selection of alpha, K, the low-cost reference and the learned checkpoint on validation suppression,
one learned model shared by all motes and trained on at most 4,000 training windows (rng seed 0),
standardisation from the training windows, seeds 0-19, checkpoints every 25 epochs up to 200, the
per-checkpoint Keras equivalence check of the fast forward pass, and the strict independent-receiver
check on the test span. Training windows never cross a segment boundary.

Usage: python3 intel_sens_run.py <obs|short|short100> cheap
       python3 intel_sens_run.py <obs|short|short100> <lightweight|cnn_lstm> <s0> <s1>
Outputs: sens_<variant>_cheap.json, sens_<variant>_<model>.json (one entry per seed, written
atomically after each seed; seeds already present are skipped), and per-seed test cell records
cells/<variant>_<arm>[_s<seed>].npz (sent flag and error of every test reading)."""
import os, sys, json, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1"); os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np, frozen_benchmark as fb, fastfwd, segeval as se
TAU, LB, UNIT = 0.10, fb.LOOK_BACK, 1.0
MAC = {"persistence": 0, "expsm": 12, "lightweight": 208, "cnn_lstm": 99488}

def load(variant):
    S = pickle.load(open(f"intel_segments_{variant}.pkl", "rb"))
    return {sp: [s for s in S if s["span"] == sp] for sp in ("train", "val", "test")}

def save_cells(path, segs, res):
    np.savez_compressed(path, sent=np.concatenate(res["sent"]), err=np.concatenate(res["err"]).astype(np.float64))

def cell_index(segs):
    return dict(mote=np.concatenate([np.full(len(s["values"]), s["mote"]) for s in segs]),
                bin=np.concatenate([s["bins"] for s in segs]), filled=np.concatenate([s["filled"] for s in segs]),
                value=np.concatenate([s["values"] for s in segs]),
                warmup=np.concatenate([np.arange(len(s["values"])) < LB for s in segs]))

def cheap(variant, D):
    V = [s["values"] for s in D["val"]]; T = [s["values"] for s in D["test"]]
    a = max(fb.ALPHA_GRID, key=lambda a: (se.seg_dps(V, fb.make_expsm(a), TAU)["rho"], -a))
    k = max(fb.LIN_K_GRID, key=lambda k: (se.seg_dps(V, fb.make_ar(k), TAU)["rho"], -k))
    arms = {"persistence": (None, 0), "expsm": (fb.make_expsm(a), 12), "linear_extrap": (fb.make_ar(k), k)}
    val = {n: se.seg_dps(V, f, TAU)["rho"] for n, (f, _) in arms.items()}
    sel = max(val, key=val.get)
    test = {}
    os.makedirs("cells", exist_ok=True)
    for n, (f, _) in arms.items():
        r = se.seg_dps(T, f, TAU, record=True); save_cells(f"cells/{variant}_{n}.npz", D["test"], r)
        v = se.seg_verify(T, lambda f=f: f, TAU)
        test[n] = {kk: r[kk] for kk in ("rho", "rho_incl_init", "mean_err", "rmse", "max_err", "n_steps", "n_tx", "n_init_reports")}
        test[n]["receiver_verified"] = v["bound_holds"]
    np.savez_compressed(f"cells/{variant}_index.npz", **cell_index(D["test"]))
    out = dict(variant=variant, alpha=a, K=k, val=val, selected=sel, selected_macs=arms[sel][1], test=test)
    json.dump(out, open(f"sens_{variant}_cheap.json", "w"), indent=1)
    print(json.dumps({n: round(t["rho"], 4) for n, t in test.items()}), "selected", sel, "alpha", a, "K", k)

def train_windows(D):
    Xs, Ys = [], []
    for s in D["train"]:                     # segment order: mote, then time (as built)
        x = s["values"]
        for t in range(LB, len(x)): Xs.append(x[t - LB:t]); Ys.append(x[t])
    Xs, Ys = np.array(Xs), np.array(Ys)
    rng = np.random.default_rng(0)
    if len(Xs) > fb.MAX_WINDOWS:
        i = rng.choice(len(Xs), fb.MAX_WINDOWS, replace=False); Xs, Ys = Xs[i], Ys[i]
    return Xs, Ys

def learned(variant, D, kind, seeds):
    V = [s["values"] for s in D["val"]]; T = [s["values"] for s in D["test"]]
    Xtr, Ytr = train_windows(D)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    Xin = (Xtr - mx) / sx; Yin = (Ytr - my) / sy
    if kind == "cnn_lstm": Xin = Xin[..., None]
    def wrap(m):
        fastfwd.verify_against_keras(m, kind); P = fastfwd.extract(m, kind)
        return lambda W: fastfwd.forward(P, (np.asarray(W, float) - mx) / sx) * sy + my
    out = f"sens_{variant}_{kind}.json"
    for sd in seeds:
        rows = json.load(open(out))["entries"] if os.path.exists(out) else []
        if any(r["predictor"] == kind and r["seed"] == sd for r in rows): continue
        t0 = time.time(); m = fb.build(kind, sd); best = (-1, None, None)
        for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP + 1, fb.CKPT_EVERY):
            m.fit(Xin, Yin, epochs=fb.CKPT_EVERY, batch_size=64, verbose=0, validation_split=0.15)
            r = se.seg_dps(V, wrap(m), TAU)["rho"]
            if r > best[0]: best = (r, ep, [w.copy() for w in m.get_weights()])
        m.set_weights(best[2]); f = wrap(m)
        r = se.seg_dps(T, f, TAU, record=True); save_cells(f"cells/{variant}_{kind}_s{sd}.npz", D["test"], r)
        v = se.seg_verify(T, lambda: f, TAU)
        rows = json.load(open(out))["entries"] if os.path.exists(out) else []      # re-read: another process may have written
        rows.append(dict(predictor=kind, seed=int(sd), selected_epoch=best[1], val_rho=best[0],
                         **{"test_" + kk: r[kk] for kk in ("rho", "rho_incl_init", "mean_err", "rmse", "max_err", "n_steps", "n_tx", "n_init_reports", "n_nonfinite_forecasts")},
                         receiver_verified=v["bound_holds"], train_windows=int(len(Xtr)), secs=round(time.time() - t0, 1)))
        tmp = out + f".tmp{os.getpid()}"
        json.dump({"variant": variant, "entries": rows}, open(tmp, "w"), indent=1); os.replace(tmp, out)
        os.makedirs("weights_sens", exist_ok=True); np.savez(f"weights_sens/{variant}_{kind}_s{sd}.npz", *m.get_weights())
        print(f"  {variant} {kind:10s} seed {sd:2d} ep{best[1]:3d} val={best[0]:.4f} test={r['rho']:.4f} verified={v['bound_holds']} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    variant, what = sys.argv[1], sys.argv[2]
    D = load(variant)
    if what == "cheap": cheap(variant, D)
    else: learned(variant, D, what, range(int(sys.argv[3]), int(sys.argv[4])))
