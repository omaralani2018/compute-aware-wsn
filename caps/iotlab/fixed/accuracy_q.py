"""PREREG13 accuracy: closed-loop validation and test suppression of the integer lightweight model, 11 streams x 20 seeds.
First checks that the float path reproduces the stored float test_rho exactly; then swaps in the bit-exact emulator."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(SC, "adaptA")); sys.path.insert(0, HERE)
import numpy as np, oracle_switch as O, qemu as Q
fb = O.fb; LB, TAU = O.LB, O.TAU
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly", "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
RUNW = os.path.join(SC, "stage17/run/weights")
def dps(M, f, u): return fb.dps(M, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=u)["rho"]
res = {}
for ds in DS:
    os.chdir(O.RUN); A, unit = O.load(ds); os.chdir(HERE)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    VAL, TEST = A[tr - LB:va], A[va - LB:]
    Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    fi = Q.in_frac(np.nanmax(np.abs(A))); norm = Q.Norm(mx, sx, my, sy, fi)
    stored = {e["seed"]: e for e in json.load(open(os.path.join(O.RUN, f"learned_{ds}.json")))["entries"] if e["predictor"] == "lightweight"}
    r = {"fi": fi, "seeds": {}}
    for s in range(20):
        wf = os.path.join(SC, "adaptA/weights", f"{ds}_lightweight_s{s}.npz")
        if not os.path.exists(wf): wf = os.path.join(RUNW, f"{ds}_lightweight_s{s}.npz")
        w = np.load(wf); P = [w["arr_%d" % i] for i in range(4)]
        if s in (0, 19):
            ff = O.learned_arm(A, "lightweight", wf); chk = dps(TEST, ff, unit)
            assert abs(chk - stored[s]["test_rho"]) < 1e-9, (ds, chk, stored[s]["test_rho"])
        fq = Q.forecaster(Q.LightQ(P, norm), fi)
        r["seeds"][s] = dict(val_float=stored[s]["val_rho"], test_float=stored[s]["test_rho"], val_q=dps(VAL, fq, unit), test_q=dps(TEST, fq, unit))
    d = [v["test_q"] - v["test_float"] for v in r["seeds"].values()]
    r["median_delta_test"] = float(np.median(d)); r["min_delta_test"] = float(min(d)); r["max_delta_test"] = float(max(d))
    res[ds] = r
    print("%-17s fi=%2d  test rho fixed minus float: median %+.4f  range [%+.4f, %+.4f]" % (ds, fi, r["median_delta_test"], r["min_delta_test"], r["max_delta_test"]), flush=True)
json.dump(res, open(os.path.join(HERE, "accuracy_q.json"), "w"), indent=1)
