"""Closed-loop test-span suppression of the fixed-point models (bit-exact emulator of
model_q15.c) against the floating-point models, with the deployed standardisation
constants (frozen_benchmark.run_dataset: windows(A, A, 12, tr, default_rng(0))).
Version 1.4. Inside run/ (needs weights/ and m0/qmodel.py):
    python3 quant_closed_loop.py STREAM KIND   -> qcl_<stream>_<kind>.json"""
import sys, json, glob, os, numpy as np
SRC = ""
sys.path.insert(0, "m0")
import frozen_benchmark as fb, fastfwd, nab_data
from dual_prediction_fix import dual_prediction_suppress as dps
import qmodel
ds, kind = sys.argv[1], sys.argv[2]
if ds == "intel_time": A, unit = np.load("intel_time.npy"), 1.0
else:
    import pandas as pd
    A, unit = nab_data.load(ds)
A = np.asarray(A, float); T = A.shape[0]; tr, va = int(0.5 * T), int(0.7 * T)
TEST = A[va - 12:]
Xtr, Ytr = fb.windows(A, A, 12, tr, np.random.default_rng(0))
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
stored = {e["seed"]: e["test_rho"] for e in json.load(open(f"learned_{ds}.json"))["entries"] if e["predictor"] == kind}
out = []
for f in sorted(glob.glob(f"weights/{ds}_{kind}_s*.npz"), key=lambda p: int(p.split("_s")[-1][:-4])):
    sd = int(f.split("_s")[-1][:-4]); Wn = np.load(f); W = [Wn[f"arr_{i}"] for i in range(len(Wn.files))]
    P = (kind,) + tuple(W)
    ff = lambda Wh: fastfwd.forward(P, (np.asarray(Wh) - mx) / sx) * sy + my
    dev = max(np.abs(Xtr - mx).max(), np.abs(Ytr - my).max())
    emu = qmodel.Emu(qmodel.quantise(kind, W), mx, 1.0 / sx, my, sy, dev)
    fq = lambda Wh: emu.fromint(emu(emu.toint(Wh)))
    rf = dps(TEST, ff, tau=0.10, look_back=12, train_frac=1e-6, unit=unit)
    rq = dps(TEST, fq, tau=0.10, look_back=12, train_frac=1e-6, unit=unit)
    # largest single-step output difference on the float closed-loop history is not tracked; report rho
    out.append(dict(seed=sd, stored=stored[sd], float=rf["rho"], fixed=rq["rho"], diff=rq["rho"] - rf["rho"],
                    maxacc=emu.maxacc, px=emu.px, fixed_max_err_over_tau=rq["max_err"] / (0.10 * unit)))
    print(f"{ds} {kind} s{sd:2d} stored {stored[sd]:.4f} float {rf['rho']:.4f} fixed {rq['rho']:.4f} diff {rq['rho']-rf['rho']:+.4f} maxacc {emu.maxacc:.3g}", flush=True)
json.dump(out, open(f"qcl_{ds}_{kind}.json", "w"), indent=1)
d = np.array([o["diff"] for o in out]); s = np.array([o["float"] - o["stored"] for o in out])
print(f"SUMMARY {ds} {kind}: n={len(d)} fixed-float diff mean {d.mean():+.4f} range {d.min():+.4f}..{d.max():+.4f}; float-vs-stored max |{np.abs(s).max():.4f}|")
