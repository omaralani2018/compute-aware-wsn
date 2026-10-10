"""Version 1.4. The untuned-baseline numbers quoted in Section 7 (paragraph 2) of the paper, with the current code.
Inside run/, next to intel_time.npy and the NAB files (README setup):
    python3 untuned_check.py               -> untuned_check.json

Section 7 says that tuning exponential smoothing raised its suppression by up to +0.10 over a fixed alpha = 0.9
(on rds_cpu_util), and that tuning the extrapolation window on taxi raised it from 0.115, with the window fixed at
K = 12, to 0.429. Both compare, on the test span at tau = 0.10, the arm with a fixed parameter against the same arm
with its validation-selected parameter (the selection of ms_numbers.py). The fixed-parameter values were first
produced by an earlier version of frozen_benchmark.py (results/bench_baselines.json, fixed alpha = 0.9 and K = 12);
this script recomputes them with the current evaluator, spans and grids, exactly as ms_numbers.py evaluates the
cheap arms."""
import json, numpy as np, frozen_benchmark as fb, nab_data as nd

T = 0.10
def spans(A):                                   # as in ms_numbers.py
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A; n = len(A); tr, va = int(.5 * n), int(.7 * n)
    return A[:tr], A[tr - fb.LOOK_BACK:va], A[va - fb.LOOK_BACK:]
def dps(M, f, u): return fb.dps(M, f, tau=T, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=u)["rho"]

M = json.load(open("ms_numbers.json"))["datasets"]
data = {"intel_time": (np.load("intel_time.npy"), 1.0)}
for k in nd.NAB: data[k] = nd.load(k)
out = dict(tau=T, fixed_alpha=0.9, fixed_K=12, streams={})
for ds, (A, u) in data.items():
    _, VAL, TEST = spans(A); b = M[ds]["by_tau"]["0.1"]
    a = max(fb.ALPHA_GRID, key=lambda x: (dps(VAL, fb.make_expsm(x), u), -x))
    k = max(fb.LIN_K_GRID, key=lambda x: (dps(VAL, fb.make_ar(x), u), -x))
    assert a == b["alpha"] and k == b["K"], ds                       # same selection as ms_numbers.json
    e_sel, l_sel = dps(TEST, fb.make_expsm(a), u), dps(TEST, fb.make_ar(k), u)
    assert e_sel == b["test"]["expsm"]["rho"] and l_sel == b["test"]["linear_extrap"]["rho"], ds
    e_fix, l_fix = dps(TEST, fb.make_expsm(0.9), u), dps(TEST, fb.make_ar(12), u)
    out["streams"][ds] = dict(alpha_selected=a, expsm_selected=e_sel, expsm_alpha_0_9=e_fix, expsm_gain=e_sel - e_fix,
                              K_selected=k, linear_selected=l_sel, linear_K_12=l_fix, linear_gain=l_sel - l_fix)
g = {ds: r["expsm_gain"] for ds, r in out["streams"].items()}
out["largest_expsm_gain"] = dict(stream=max(g, key=g.get), gain=max(g.values()))
json.dump(out, open("untuned_check.json", "w"), indent=1)
print(out["largest_expsm_gain"], {k: round(v, 4) for k, v in out["streams"]["taxi"].items() if "linear" in k or k == "K_selected"})
