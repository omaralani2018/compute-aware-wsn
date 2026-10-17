"""Grand St Bernard run (STB_PROTOCOL.md, sections 4 and 6), on one preparation from stb_prepare.py,
evaluated on independent per-station segments with segeval.py.

Pooling: every suppression used for selection or reporting is pooled over all closed-loop steps of
all segments of the span (1 - transmissions / steps), not a mean of station rates.

Usage:  python3 stb_run.py <obs|fill> cheap
        python3 stb_run.py <obs|fill> <lightweight|cnn_lstm> <s0> <s1>
Outputs (in the run directory):
  stb_<prep>_cheap.json                 low-cost arms: alpha, K, validation and test results, reference
  runs/stb_<prep>_<model>_s<seed>.json  one file per learned seed (status ok or failed)
  cells/stb_<prep>_index.npz            station, time, filled and initialisation flag of every test reading
  cells/stb_<prep>_<arm>[_s<seed>].npz  sent flag and error of every test reading
  weights_stb/<prep>_<model>_s<seed>.npz
A seed is attempted at most twice with the same seed: an exception inside the process is retried
at once, and a process that dies is restarted by stb_driver.sh, which runs one seed per process; the
number of attempts is kept in runs/<name>.attempts. After two failed attempts the seed is written as
failed and the run continues with the next seed. A preparation marked not evaluable in
stb_meta.json is not run."""
import os, sys, json, time, pickle, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1"); os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np, frozen_benchmark as fb, fastfwd, segeval as se
TAU, LB = 0.10, fb.LOOK_BACK

def load(prep):
    S = pickle.load(open(f"stb_segments_{prep}.pkl", "rb"))
    return {sp: [s for s in S if s["span"] == sp] for sp in ("train", "val", "test")}

def vals(segs): return [s["values"] for s in segs]

def save_cells(path, res):
    np.savez_compressed(path, sent=np.concatenate(res["sent"]), err=np.concatenate(res["err"]).astype(np.float64))

def cheap(prep, D):
    V, T = vals(D["val"]), vals(D["test"])
    a = max(fb.ALPHA_GRID, key=lambda a: (se.seg_dps(V, fb.make_expsm(a), TAU)["rho"], -a))
    k = max(fb.LIN_K_GRID, key=lambda k: (se.seg_dps(V, fb.make_ar(k), TAU)["rho"], -k))
    make = {"persistence": (lambda: None), "expsm": (lambda: fb.make_expsm(a)), "linear_extrap": (lambda: fb.make_ar(k))}
    macs = {"persistence": 0, "expsm": 12, "linear_extrap": k}
    val = {n: se.seg_dps(V, mk(), TAU)["rho"] for n, mk in make.items()}
    sel = max(val, key=val.get)                       # dict order breaks ties: persistence, smoothing, extrapolation
    os.makedirs("cells", exist_ok=True)
    np.savez_compressed(f"cells/stb_{prep}_index.npz",
        mote=np.concatenate([np.full(len(s["values"]), s["mote"]) for s in D["test"]]),
        bin=np.concatenate([s["bins"] for s in D["test"]]), filled=np.concatenate([s["filled"] for s in D["test"]]),
        value=np.concatenate([s["values"] for s in D["test"]]), count=np.concatenate([s["count"] for s in D["test"]]),
        warmup=np.concatenate([np.arange(len(s["values"])) < LB for s in D["test"]]))
    test = {}
    for n, mk in make.items():
        r = se.seg_dps(T, mk(), TAU, record=True); save_cells(f"cells/stb_{prep}_{n}.npz", r)
        v = se.seg_verify(T, mk, TAU)                 # sender and receiver get separate predictor objects
        test[n] = {kk: r[kk] for kk in ("rho", "rho_incl_init", "mean_err", "rmse", "max_err", "n_steps", "n_tx", "n_init_reports")}
        test[n].update(receiver_verified=v["bound_holds"], sender_receiver_identical=v["sender_receiver_identical"], macs=macs[n])
    out = dict(prep=prep, alpha=a, K=k, val=val, selected=sel, selected_macs=macs[sel], test=test)
    json.dump(out, open(f"stb_{prep}_cheap.json", "w"), indent=1)
    print(json.dumps({n: round(t["rho"], 4) for n, t in test.items()}), "selected", sel, "alpha", a, "K", k, flush=True)

def train_windows(D):
    Xs, Ys = [], []
    for s in D["train"]:
        x = s["values"]
        for t in range(LB, len(x)): Xs.append(x[t - LB:t]); Ys.append(x[t])
    Xs, Ys = np.array(Xs), np.array(Ys)
    rng = np.random.default_rng(0)
    if len(Xs) > fb.MAX_WINDOWS:
        i = rng.choice(len(Xs), fb.MAX_WINDOWS, replace=False); Xs, Ys = Xs[i], Ys[i]
    return Xs, Ys

def one_seed(prep, D, kind, sd, Xtr, Ytr):
    V, T = vals(D["val"]), vals(D["test"])
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    Xin = (Xtr - mx) / sx; Yin = (Ytr - my) / sy
    if kind == "cnn_lstm": Xin = Xin[..., None]
    def wrap(m):
        fastfwd.verify_against_keras(m, kind); P = fastfwd.extract(m, kind)
        return lambda W: fastfwd.forward(P, (np.asarray(W, float) - mx) / sx) * sy + my
    t0 = time.time(); m = fb.build(kind, sd); best = (-1.0, None, None)
    for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP + 1, fb.CKPT_EVERY):
        m.fit(Xin, Yin, epochs=fb.CKPT_EVERY, batch_size=64, verbose=0, validation_split=0.15)
        r = se.seg_dps(V, wrap(m), TAU)["rho"]
        if r > best[0]: best = (r, ep, [w.copy() for w in m.get_weights()])
    m.set_weights(best[2])
    r = se.seg_dps(T, wrap(m), TAU, record=True); save_cells(f"cells/stb_{prep}_{kind}_s{sd}.npz", r)
    v = se.seg_verify(T, lambda: wrap(m), TAU)
    os.makedirs("weights_stb", exist_ok=True); np.savez(f"weights_stb/{prep}_{kind}_s{sd}.npz", *m.get_weights())
    return dict(status="ok", predictor=kind, seed=int(sd), selected_epoch=best[1], val_rho=best[0],
                **{"test_" + kk: r[kk] for kk in ("rho", "rho_incl_init", "mean_err", "rmse", "max_err", "n_steps", "n_tx", "n_init_reports", "n_nonfinite_forecasts")},
                receiver_verified=v["bound_holds"], sender_receiver_identical=v["sender_receiver_identical"],
                train_windows=int(len(Xtr)), secs=round(time.time() - t0, 1))

def learned(prep, D, kind, seeds):
    os.makedirs("runs", exist_ok=True); os.makedirs("cells", exist_ok=True)
    Xtr, Ytr = train_windows(D)
    for sd in seeds:
        path = f"runs/stb_{prep}_{kind}_s{sd}.json"; mark = path + ".attempts"
        if os.path.exists(path): continue
        res, errs = None, []
        done = int(open(mark).read()) if os.path.exists(mark) else 0      # attempts already started (a process may have died)
        while res is None and done < 2:
            done += 1; open(mark, "w").write(str(done))
            try:
                res = one_seed(prep, D, kind, sd, Xtr, Ytr); res["attempts"] = done
            except Exception:
                errs.append(traceback.format_exc(limit=3))
        if res is None: res = dict(status="failed", predictor=kind, seed=int(sd), attempts=done, errors=errs)
        json.dump(res, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
        print(f"  stb {prep} {kind:10s} seed {sd:2d} {res['status']}" + (f" ep{res['selected_epoch']:3d} val={res['val_rho']:.4f} test={res['test_rho']:.4f} verified={res['receiver_verified']} {res['secs']:.0f}s" if res["status"] == "ok" else ""), flush=True)

if __name__ == "__main__":
    prep, what = sys.argv[1], sys.argv[2]
    meta = json.load(open("stb_meta.json"))
    if not meta["evaluable"].get(prep, False):
        print(f"preparation {prep} is not evaluable: {meta['not_evaluable_reason']}"); sys.exit(0)
    D = load(prep)
    if what == "cheap": cheap(prep, D)
    else: learned(prep, D, what, range(int(sys.argv[3]), int(sys.argv[4])))
