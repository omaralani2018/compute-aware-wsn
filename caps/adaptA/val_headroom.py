import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import oracle_switch as O
fb = O.fb; out = {}
for ds in ["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]:
    A, unit = O.load(ds); A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr, va = int(fb.SPLIT[0]*T), int(fb.SPLIT[1]*T); X = A[tr-O.LB:va]; n = (X.shape[0]-O.LB)*X.shape[1]
    _, fR, mR, rvalR, _ = O.ref_arm(ds)
    txR = O.static(X, fR, unit).sum(); assert abs((1-txR/n) - rvalR) < 1e-9, (ds, 1-txR/n, rvalR)
    ER_ = txR*O.ER + n*mR*O.EOP; G = []
    for sd in range(20):
        wf = next(p for p in [os.path.join(HERE,"weights",f"{ds}_lightweight_s{sd}.npz"), os.path.join(O.RUN,"weights",f"{ds}_lightweight_s{sd}.npz")] if os.path.exists(p))
        fL = O.learned_arm(A, "lightweight", wf); txL = O.static(X, fL, unit).sum(); EL = txL*O.ER + n*208*O.EOP
        Es = O.oracle(X, [fR, fL], [mR, 208], 96, unit)[0]; G.append(1 - Es/min(ER_, EL))
    test = json.load(open(os.path.join(HERE, f"oracle_{ds}_lightweight.json")))["rows"]
    gt = float(np.median([r["B96"]["G_best"] for r in test])); gv = float(np.median(G))
    out[ds] = {"val_headroom": gv, "test_headroom": gt, "flag_val": gv >= 0.05, "flag_test": gt >= 0.05}
    print("%-17s val=%6.2f%%  test=%6.2f%%  %s" % (ds, 100*gv, 100*gt, "agree" if (gv>=0.05)==(gt>=0.05) else "DISAGREE"), flush=True)
agree = sum(v["flag_val"] == v["flag_test"] for v in out.values()); print("agreement", agree, "/ 11")
json.dump({"streams": out, "agreement": agree}, open(os.path.join(HERE, "val_headroom.json"), "w"), indent=1)
