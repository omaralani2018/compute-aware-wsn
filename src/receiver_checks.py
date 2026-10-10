"""Independent-receiver verification of the cheap arms (persistence, exponential smoothing,
linear extrapolation) at every tolerance of the paper (0.05, 0.10, 0.20, 0.50), on the
validation and test spans of all eleven streams, with the parameters (alpha, K) that
ms_numbers.py selected on validation. verify_with_independent_receiver runs in strict mode,
so any divergence between sender and receiver, or any receiver error above tau, raises.
The suppression it reports is also checked against ms_numbers.json for both spans.
Inputs: ms_numbers.json, intel_time.npy and the NAB files (nab_repo/data/...).
Output: receiver_checks.json."""
import json, numpy as np, frozen_benchmark as fb, nab_data as nd
from dual_prediction_fix import verify_with_independent_receiver as verify
N = json.load(open("ms_numbers.json"))["datasets"]
data = {"intel_time": (np.load("intel_time.npy"), 1.0)}
for k in nd.NAB: data[k] = nd.load(k)
def spans(A):                                   # identical to ms_numbers.spans
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A; T = len(A); tr, va = int(.5*T), int(.7*T)
    return A[tr-fb.LOOK_BACK:va], A[va-fb.LOOK_BACK:]
runs, mismatches = [], []
for ds, (A, u) in data.items():
    VAL, TEST = spans(A)
    for t, b in N[ds]["by_tau"].items():
        arms = {"persistence": None, "expsm": fb.make_expsm(b["alpha"]), "linear_extrap": fb.make_ar(b["K"])}
        for name, f in arms.items():
            for span, X in (("val", VAL), ("test", TEST)):
                v = verify(X, (lambda ff: (lambda: ff))(f), tau=float(t), look_back=fb.LOOK_BACK,
                           train_frac=1e-6, unit=u, strict=True)
                ref = b["val"][name] if span == "val" else b["test"][name]["rho"]
                if abs(v["rho"] - ref) > 1e-12: mismatches.append([ds, t, name, span, v["rho"], ref])
                runs.append(dict(dataset=ds, tau=float(t), arm=name, span=span, alpha=b["alpha"], K=b["K"],
                                 rho=v["rho"], sender_receiver_identical=v["sender_receiver_identical"],
                                 max_err_at_receiver=v["max_err_at_receiver"], tau_times_unit=float(t)*u))
out = {"n_runs": len(runs), "all_passed": all(r["sender_receiver_identical"] for r in runs),
       "rho_mismatches_vs_ms_numbers": mismatches, "runs": runs}
json.dump(out, open("receiver_checks.json", "w"), indent=1)
print({k: out[k] for k in ("n_runs", "all_passed", "rho_mismatches_vs_ms_numbers")})
