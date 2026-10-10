"""Training on actual versus reconstructed history (TM_PROTOCOL.md): one (stream, seed) per process.

Phase 1 repeats the main-study training of the lightweight network exactly as run_learned.py does
(same windows, standardisation, build(seed), 8 fits of 25 epochs with Keras's 15% validation split,
the same validation scoring after each fit), so its validation-selected checkpoint reproduces the
main-study entry for that seed (checked against learned_<stream>.json when present). The model after
epoch 200, with its Adam state, is saved and both arms continue from it:
  B+  200 further epochs on the same actual-history windows;
  C   the epoch-200 model is run in closed loop over the training span (tau, shared state, as at
      deployment); the training windows are rebuilt from that reconstructed series at the same window
      positions and the same sampled indices (inputs reconstructed, targets the actual readings);
      200 further epochs on them.
The starting state is captured and saved right after the last phase-1 fit, before any validation-selected
weights are restored; the selected weights are loaded into the phase-1 model only afterwards, to score
the test span, and the epoch-200 weights are then put back for the closed-loop pass. Both arms reload the
saved state; before training, each arm's state (weights, step count and every optimiser variable) is
checked against it and the SHA-256 of the two states is recorded and required equal. Both arms then use
the same random seed, the same number of windows at the same positions (so the same Keras holdout
membership and batch order, which depends only on the seed and the number of windows) and the same
targets; their update counts are recorded and required equal. With identical windows the two arms are
identical (checked in tm_synthetic_check.py). In each arm a checkpoint every 25 epochs of the second
phase is scored on validation suppression and the highest is kept (ties: the earlier one), then the test
span is scored once with the independent-receiver check.

Usage (inside a run directory set up as in the package README):
    python3 tm_run.py <stream> <seed>          -> tm_runs/tm_<stream>_s<seed>.json
"""
import os
for _v in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[_v] = "1"                     # as run_learned.py (whose setdefault leaves these at 1 here)
import sys, json, time, shutil, hashlib, platform, numpy as np
import frozen_benchmark as fb, nab_data as nd
from run_learned import wrap_fast

KIND = "lightweight"
OUT_DIR = "tm_runs"


def data_file(ds):
    return "intel_time.npy" if ds == "intel_time" else nd.NAB.get(ds, "")


def load(ds):
    if ds == "intel_time":
        return np.load("intel_time.npy"), 1.0
    return nd.load(ds)


def reconstruct(X, f, tau, look_back, unit):
    """The shared state of dual prediction over X, the loop of dual_prediction_suppress with
    t0 = look_back: the first look_back rows are synchronised; afterwards a reading is replaced by
    the forecast whenever the forecast is within tau. Returns (reconstructed series, suppression)."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    recon = X.copy(); n_tx = 0
    for t in range(look_back, X.shape[0]):
        pred = np.asarray(f(recon[t - look_back:t, :].T), dtype=float).reshape(-1)
        assert pred.shape[0] == X.shape[1]
        pred = np.where(np.isfinite(pred), pred, np.inf)
        s = (np.abs(X[t, :] - pred) / unit) <= tau
        recon[t, :] = np.where(s, pred, X[t, :]); n_tx += int((~s).sum())
    return recon, 1.0 - n_tx / ((X.shape[0] - look_back) * X.shape[1])


def state_sha256(m):
    """SHA-256 of a model's training state: step count, weights and every optimiser variable, in order."""
    h = hashlib.sha256(str(int(m.optimizer.iterations.numpy())).encode())
    for a in list(m.get_weights()) + [np.array(v.numpy()) for v in m.optimizer.variables]:
        a = np.ascontiguousarray(a); h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def train_phase(m, Xin, Yin, VAL, unit, stds, epoch0=0):
    """8 fits of CKPT_EVERY epochs, each followed by validation scoring; returns the best checkpoint."""
    best, hist = (-1.0, None, None), []
    for ep in range(fb.CKPT_EVERY, fb.EPOCH_CAP + 1, fb.CKPT_EVERY):
        m.fit(Xin, Yin, epochs=fb.CKPT_EVERY, batch_size=64, verbose=0, validation_split=0.15)
        r = fb.dps(VAL, wrap_fast(m, KIND, *stds), tau=fb.TAU, look_back=fb.LOOK_BACK,
                   train_frac=1e-6, unit=unit)["rho"]
        hist.append(float(r))
        if r > best[0]:
            best = (float(r), epoch0 + ep, [w.copy() for w in m.get_weights()])
    return best, hist


def score_test(m, weights, TEST, unit, stds):
    m.set_weights(weights); f = wrap_fast(m, KIND, *stds)
    t_ = fb.dps(TEST, f, tau=fb.TAU, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)
    vr = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=fb.TAU, look_back=fb.LOOK_BACK,
                   train_frac=1e-6, unit=unit, strict=False)      # a failure is recorded, not raised
    return dict(test_rho=t_["rho"], test_n_tx=int(t_["n_tx"]), test_mean_err=t_["mean_err"], test_max_err=t_["max_err"],
                test_rmse=t_["rmse"], test_nonfinite=t_["n_nonfinite_forecasts"],
                test_n_steps=t_["n_steps"], test_n_init_reports=t_["n_init_reports"],
                receiver_verified=bool(vr["bound_holds"] and vr["sender_receiver_identical"]),
                receiver_bound_holds=bool(vr["bound_holds"]), receiver_identical=bool(vr["sender_receiver_identical"]),
                receiver_max_err=float(vr["max_err_at_receiver"]),
                energy=fb.energy_report(KIND, t_["rho"], t_["n_steps"], t_["n_init_reports"]))


def run(A, name, unit, seed, tau_recon=None, workdir=None, phase1_only=False):
    """phase1_only: stop after phase 1 (used to check the reproduction of the main study).
    tau_recon: tolerance of the closed-loop pass that builds the arm-C windows (default fb.TAU;
    the synthetic check passes 0.0, which makes the reconstructed history equal the actual one)."""
    from tensorflow import keras
    t0 = time.time()
    tau_recon = fb.TAU if tau_recon is None else tau_recon
    A = np.asarray(A, float)
    if A.ndim == 1: A = A[:, None]
    T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    VAL, TEST = A[tr - fb.LOOK_BACK:va], A[va - fb.LOOK_BACK:]
    Xtr, Ytr = fb.windows(A, A, fb.LOOK_BACK, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    stds = (mx, sx, my, sy)
    Xin, Yin = (Xtr - mx) / sx, (Ytr - my) / sy

    # phase 1: the main-study training of this seed
    m = fb.build(KIND, seed)
    best1, hist1 = train_phase(m, Xin, Yin, VAL, unit, stds)
    w200 = [w.copy() for w in m.get_weights()]      # epoch-200 state: nothing has been restored (train_phase only copies)
    sha200 = state_sha256(m)
    tmp = os.path.join(workdir or ".", f"tm_tmp_{name}_s{int(seed)}")   # per run: a stale copy is replaced
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    path = os.path.join(tmp, "m200.keras"); m.save(path)
    it200 = int(m.optimizer.iterations.numpy())
    opt200 = [np.array(v.numpy()) for v in m.optimizer.variables]
    phase1 = dict(selected_epoch=best1[1], val_rho=best1[0], val_hist=hist1,
                  **score_test(m, best1[2], TEST, unit, stds))
    if phase1_only:
        shutil.rmtree(tmp)
        return dict(stream=name, seed=int(seed), phase1=phase1, start_state=dict(sha256=sha200, iterations=it200))

    # arm C windows: closed loop of the epoch-200 model over the training span
    m.set_weights(w200)
    R, rho_train = reconstruct(A[:tr], wrap_fast(m, KIND, *stds), tau_recon, fb.LOOK_BACK, unit)
    Rfull = A.copy(); Rfull[:tr] = R
    Xc, Yc = fb.windows(Rfull, A, fb.LOOK_BACK, tr, np.random.default_rng(0))
    assert Xc.shape == Xtr.shape and np.array_equal(Yc, Ytr), "arm C windows must keep positions and targets"
    Xcin = (Xc - mx) / sx
    recon_info = dict(tau=tau_recon, train_closed_loop_rho=float(rho_train),
                      windows_changed=float(np.mean(np.any(Xc != Xtr, axis=1))),
                      inputs_changed=float(np.mean(Xc != Xtr)),
                      mean_abs_input_change=float(np.mean(np.abs(Xc - Xtr))))

    arms = {}
    for arm, X_arm in (("B+", Xin), ("C", Xcin)):
        m2 = keras.models.load_model(path)
        assert int(m2.optimizer.iterations.numpy()) == it200, "reloaded step count differs"
        assert all(np.array_equal(a, b) for a, b in zip(m2.get_weights(), w200)), "reloaded weights differ"
        o2 = [np.array(v.numpy()) for v in m2.optimizer.variables]            # Adam state, by position
        assert len(o2) == len(opt200) and all(np.array_equal(a, b) for a, b in zip(o2, opt200)), "reloaded optimiser state differs"
        start = state_sha256(m2); assert start == sha200, "arm start state differs from the epoch-200 state"
        assert X_arm.shape == Xin.shape, "arm windows differ in number"   # same positions and holdout membership
        keras.utils.set_random_seed(int(seed) + 1000)          # same shuffling in both arms
        best2, hist2 = train_phase(m2, X_arm, Yin, VAL, unit, stds, epoch0=fb.EPOCH_CAP)
        arms[arm] = dict(selected_epoch=best2[1], val_rho=best2[0], val_hist=hist2, start_state_sha256=start,
                         updates=int(m2.optimizer.iterations.numpy()) - it200,
                         **score_test(m2, best2[2], TEST, unit, stds))
    assert arms["B+"]["start_state_sha256"] == arms["C"]["start_state_sha256"], "arm start states differ"
    assert arms["B+"]["updates"] == arms["C"]["updates"] > 0, "arm update counts differ"
    shutil.rmtree(tmp)

    main = None
    if os.path.exists(f"learned_{name}.json"):
        E = [e for e in json.load(open(f"learned_{name}.json"))["entries"]
             if e["predictor"] == KIND and e["seed"] == int(seed)]
        if E:
            e = E[0]
            main = dict(selected_epoch=e["selected_epoch"], val_rho=e["val_rho"], test_rho=e["test_rho"],
                        reproduced=bool(e["selected_epoch"] == phase1["selected_epoch"]
                                        and e["val_rho"] == phase1["val_rho"] and e["test_rho"] == phase1["test_rho"]))
    from tensorflow import __version__ as tfv
    return dict(stream=name, seed=int(seed), kind=KIND, unit=unit, T=T, split=[tr, va], tau=fb.TAU,
                phase1=phase1, main_study_entry=main,
                start_state=dict(sha256=sha200, iterations=it200, n_windows=int(len(Xin)), validation_split=0.15,
                                 captured="after the last phase-1 fit, before any validation-selected weights are restored"), reconstruction=recon_info, arms=arms,
                d_test_rho_C_minus_Bplus=arms["C"]["test_rho"] - arms["B+"]["test_rho"],
                versions=dict(python=platform.python_version(), numpy=np.__version__, tensorflow=tfv,
                              keras=keras.__version__),
                settings=dict(LOOK_BACK=fb.LOOK_BACK, TAU=fb.TAU, SPLIT=list(fb.SPLIT), EPOCH_CAP=fb.EPOCH_CAP,
                              CKPT_EVERY=fb.CKPT_EVERY, MAX_WINDOWS=fb.MAX_WINDOWS, phase2_seed=int(seed) + 1000,
                              threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
                              data_sha256=hashlib.sha256(open(data_file(name), "rb").read()).hexdigest()
                              if data_file(name) and os.path.exists(data_file(name)) else None,
                              code_sha256={f: hashlib.sha256(open(f, "rb").read()).hexdigest()
                                           for f in ("tm_run.py", "frozen_benchmark.py", "run_learned.py", "fastfwd.py",
                                                     "dual_prediction_fix.py", "nab_data.py") if os.path.exists(f)}),
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    if sys.argv[1] == "--check":            # setup check used by tm_driver.sh before any attempt
        import tensorflow as tf, keras
        want = dict(python="3.11", numpy="2.4.4", tensorflow="2.21.0", keras="3.15.1")
        have = dict(python="%d.%d" % sys.version_info[:2], numpy=np.__version__, tensorflow=tf.__version__,
                    keras=keras.__version__)
        assert have == want, ("software versions differ from the protocol", have)
        print("setup ok: versions", have)
        for ds in sys.argv[2:]:
            A, unit = load(ds); assert np.isfinite(A).all() and unit > 0, ds
            print(f"setup ok: {ds} T={len(A)} N={A.shape[1] if A.ndim > 1 else 1} unit={unit:.6g}")
        sys.exit(0)
    ds, sd = sys.argv[1], int(sys.argv[2])
    A, unit = load(ds)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"tm_{ds}_s{sd}.json")
    res = run(A, ds, unit, sd, workdir=".")
    tmpf = out + ".tmp"; json.dump(res, open(tmpf, "w"), indent=1); os.replace(tmpf, out)
    print(f"[{ds} s{sd}] phase1 ep{res['phase1']['selected_epoch']} test={res['phase1']['test_rho']:.4f} "
          f"repro={None if res['main_study_entry'] is None else res['main_study_entry']['reproduced']} | "
          f"B+ {res['arms']['B+']['test_rho']:.4f}  C {res['arms']['C']['test_rho']:.4f}  "
          f"d={res['d_test_rho_C_minus_Bplus']:+.4f}  {res['secs']:.0f}s", flush=True)
