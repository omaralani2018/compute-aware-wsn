"""Version 1.6. Independent-receiver check of the closed-loop runs of the integer code (Section 6.14) with the first
emulator, qemu_v1_si20.py (fixed 20-bit input scale), 11 streams x 20 seeds; also checks that each run's test
suppression equals iotlab/fixed/accuracy_q_v1_fixed_si20.json. Prints the number of runs and of failures.
Run in caps/recv after setup.sh."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); FX = os.path.join(SC, "iotlab/fixed")
sys.path.insert(0, os.path.join(SC, "adaptA")); sys.path.insert(0, FX)
import numpy as np, oracle_switch as O, qemu_v1_si20 as Q
fb = O.fb; LB, TAU = O.LB, O.TAU
ACC = json.load(open(os.path.join(FX, "accuracy_q_v1_fixed_si20.json")))
RUNW = os.path.join(SC, "stage17/run/weights"); here = os.getcwd(); bad = 0; n = 0
for ds in ACC:
    os.chdir(O.RUN); A, unit = O.load(ds); os.chdir(here)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T); TEST = A[va - LB:]
    Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    fi = Q.in_frac(np.nanmax(np.abs(A))); norm = Q.Norm(mx, sx, my, sy, fi)
    for s in range(20):
        wf = os.path.join(SC, "adaptA/weights", f"{ds}_lightweight_s{s}.npz")
        if not os.path.exists(wf): wf = os.path.join(RUNW, f"{ds}_lightweight_s{s}.npz")
        w = np.load(wf); P = [w["arr_%d" % i] for i in range(4)]
        fq = Q.forecaster(Q.LightQ(P, norm), fi)
        vr = fb.verify(TEST, (lambda ff: (lambda: ff))(fq), tau=TAU, look_back=LB, train_frac=1e-6, unit=unit, strict=True)
        rho = fb.dps(TEST, fq, tau=TAU, look_back=LB, train_frac=1e-6, unit=unit)["rho"]
        n += 1; ok = vr["bound_holds"] and abs(rho - ACC[ds]["seeds"][str(s)]["test_q"]) < 1e-12
        bad += (not ok)
    print(ds, "done", flush=True)
print("runs", n, "failed", bad)
