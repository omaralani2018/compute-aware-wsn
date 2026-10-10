"""seg_dps reproduces the main-study evaluator on the published Intel matrix: the 30 columns of the
validation and test spans (each with its 12 warm-up rows) are 30 segments, and every arm's
suppression and errors must equal fb.dps on the matrix; the shipped CNN-LSTM and lightweight
weights of seed 0 must reproduce the shipped test suppression. Also checks seg_verify."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, numpy as np, frozen_benchmark as fb, fastfwd, segeval as se
A = np.load("intel_time.npy"); T = len(A); tr, va = int(.5 * T), int(.7 * T); LB = fb.LOOK_BACK
VAL, TEST = A[tr - LB:va], A[va - LB:]
M = json.load(open("ms_numbers.json"))["datasets"]["intel_time"]["by_tau"]["0.1"]
arms = {"persistence": None, "expsm": fb.make_expsm(M["alpha"]), "linear_extrap": fb.make_ar(M["K"])}
for n, f in arms.items():
    for name, S in (("val", VAL), ("test", TEST)):
        a = fb.dps(S, f, tau=0.1, look_back=LB, train_frac=1e-6, unit=1.0)
        b = se.seg_dps([S[:, j] for j in range(S.shape[1])], f, 0.1)
        assert a["n_tx"] == b["n_tx"] and a["n_steps"] == b["n_steps"] and a["n_init_reports"] == b["n_init_reports"], (n, name)
        # decisions identical; error metrics equal to rounding (memory layout of the forecast windows differs)
        assert all(abs(a[k] - b[k]) < 1e-10 for k in ("mean_err", "max_err", "rmse")), (n, name)
    assert abs(b["rho"] - M["test"][n]["rho"]) < 1e-15
    v = se.seg_verify([TEST[:, j] for j in range(30)], lambda f=f: f, 0.1)
    assert v["bound_holds"] and v["sender_receiver_identical"]
    print(n, "identical; receiver ok; test rho", b["rho"])
# learned, seed 0, from shipped weights
rng = np.random.default_rng(0)
Xtr, Ytr = fb.windows(A, A, LB, tr, rng)
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
L = {(r["predictor"], r["seed"]): r for r in json.load(open("learned_intel_time.json"))["entries"]}
for kind in ("lightweight", "cnn_lstm"):
    m = fb.build(kind, 0); w = np.load(f"weights/intel_time_{kind}_s0.npz"); m.set_weights([w[k] for k in w.files])
    P = fastfwd.extract(m, kind)
    f = lambda W, P=P: fastfwd.forward(P, (np.asarray(W, float) - mx) / sx) * sy + my
    b = se.seg_dps([TEST[:, j] for j in range(30)], f, 0.1)
    a = fb.dps(TEST, f, tau=0.1, look_back=LB, train_frac=1e-6, unit=1.0)
    assert a["n_tx"] == b["n_tx"] and abs(a["mean_err"] - b["mean_err"]) < 1e-10
    print(kind, "seg", b["rho"], "fb", a["rho"], "shipped", L[(kind, 0)]["test_rho"])
    assert abs(b["rho"] - L[(kind, 0)]["test_rho"]) < 1e-12
print("ALL OK")
