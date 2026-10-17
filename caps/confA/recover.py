"""Rebuild per-seed records lost in a write conflict from the saved weights (best checkpoint of each seed).
Method check first: records that survived are recomputed and must match exactly."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../adaptA")); sys.path.insert(0, HERE)
import numpy as np, oracle_switch as O, conf_data as cd
fb = O.fb; LB, TAU = fb.LOOK_BACK, fb.TAU
for ds in cd.FILES:
    A, u = cd.load(ds); T = len(A); tr, va = int(.5 * T), int(.7 * T); VAL, TEST = A[tr - LB:va], A[va - LB:]
    p = os.path.join(HERE, f"learned_{ds}.json"); J = json.load(open(p)); have = {(e["predictor"], e["seed"]): e for e in J["entries"]}
    for kind in ("lightweight", "cnn_lstm"):
        for s in range(20):
            f = O.learned_arm(A, kind, os.path.join(HERE, "weights", f"{ds}_{kind}_s{s}.npz"))
            v = fb.dps(VAL, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=u)["rho"]
            t = fb.dps(TEST, f, tau=TAU, look_back=LB, train_frac=1e-6, unit=u)
            if (kind, s) in have:
                e = have[(kind, s)]; assert abs(e["val_rho"] - v) < 1e-12 and abs(e["test_rho"] - t["rho"]) < 1e-12, (ds, kind, s, e["val_rho"], v, e["test_rho"], t["rho"])
            else:
                vr = fb.verify(TEST, (lambda ff: (lambda: ff))(f), tau=TAU, look_back=LB, train_frac=1e-6, unit=u, strict=True)
                J["entries"].append({"predictor": kind, "seed": s, "selected_epoch": None, "val_rho": v, "test_rho": t["rho"],
                                     "test_max_err": t["max_err"], "receiver_verified": vr["bound_holds"], "recovered_from_weights": True})
                print("recovered", ds, kind, s, round(v, 4), round(t["rho"], 4), flush=True)
    json.dump(J, open(p, "w"), indent=1)
print("all surviving records reproduced exactly; recovery done")
