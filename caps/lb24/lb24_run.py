"""PREREG9: look-back 24. Usage: python3 lb24_run.py <ds> cheap|lightweight"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
sys.path.insert(0, RUN); os.chdir(HERE)
import numpy as np
import frozen_benchmark as fb
fb.LOOK_BACK = 24; fb.LIN_K_GRID = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24]; fb.MACS["expsm"] = 24; fb.MACS["lightweight"] = 400
import run_learned as rl, nab_data as nd
ds, what = sys.argv[1], sys.argv[2]
A, unit = (np.load("intel_time.npy"), 1.0) if ds == "intel_time" else nd.load(ds)
A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
if what == "cheap":
    T = len(A); tr, va = int(.5 * T), int(.7 * T); LB = fb.LOOK_BACK
    VAL, TEST = A[tr - LB:va], A[va - LB:]
    d = lambda M, f: fb.dps(M, f, tau=fb.TAU, look_back=LB, train_frac=1e-6, unit=unit)
    a = max(fb.ALPHA_GRID, key=lambda a: (d(VAL, fb.make_expsm(a))["rho"], -a))
    k = max(fb.LIN_K_GRID, key=lambda k: (d(VAL, fb.make_ar(k))["rho"], -k))
    arms = {"persistence": (None, 0), "expsm": (fb.make_expsm(a), 24), "linear_extrap": (fb.make_ar(k), k)}
    val = {n: d(VAL, f)["rho"] for n, (f, _) in arms.items()}; sel = max(val, key=val.get)
    test = {n: d(TEST, f)["rho"] for n, (f, _) in arms.items()}
    json.dump(dict(alpha=a, K=k, val=val, test=test, selected=sel, selected_macs=arms[sel][1]), open(f"cheap_{ds}.json", "w"), indent=1)
    print(ds, "cheap", sel, round(test[sel], 4), flush=True)
else:
    assert fb.build("lightweight", 0).input_shape[1] == 24
    rl.run(A, ds, unit, ["lightweight"], range(20), f"learned_{ds}.json")
