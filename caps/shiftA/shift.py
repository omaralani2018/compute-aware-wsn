"""PREREG8: open-loop vs closed-loop test suppression for MSE, TA and CLT lightweight models."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "adaptA"))
import numpy as np
from scipy.stats import wilcoxon, spearmanr
import oracle_switch as O
fb = O.fb; LB, TAU = O.LB, O.TAU
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly", "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
RUNW = os.path.join(SC, "stage17/run/weights")

def wfile(kind, ds, s):
    if kind == "mse":
        p = os.path.join(SC, "adaptA/weights", f"{ds}_lightweight_s{s}.npz")
        return p if os.path.exists(p) else os.path.join(RUNW, f"{ds}_lightweight_s{s}.npz")
    return os.path.join(SC, {"ta": "tolA", "clt": "cltA"}[kind], "weights", f"{ds}_{kind}_s{s}.npz")

def closed(kind, ds):
    if kind == "mse": f, key = os.path.join(SC, "stage17/run", f"learned_{ds}.json"), None
    else: f = os.path.join(SC, {"ta": "tolA", "clt": "cltA"}[kind], f"{kind}_{ds}.json")
    E = json.load(open(f))["entries"]
    if kind == "mse": E = [e for e in E if e["predictor"] == "lightweight"]
    return {e["seed"]: e["test_rho"] for e in E}

def holm(p, a=0.05):
    o = np.argsort(p); rej = np.zeros(len(p), bool)
    for i, k in enumerate(o):
        if p[k] <= a / (len(p) - i): rej[k] = True
        else: break
    return rej

res = {}
for ds in DS:
    os.chdir(O.RUN); A, unit = O.load(ds); os.chdir(HERE)
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; va = int(fb.SPLIT[1] * T); X = A[va - LB:]
    W = np.concatenate([np.stack([X[t - LB:t, j] for t in range(LB, X.shape[0])]) for j in range(X.shape[1])])
    Y = np.concatenate([X[LB:, j] for j in range(X.shape[1])])
    r = {}
    for kind in ("mse", "ta", "clt"):
        cl = closed(kind, ds); op = {}
        for s in range(20):
            f = O.learned_arm(A, "lightweight", wfile(kind, ds, s))
            op[s] = float(np.mean(np.abs(np.asarray(f(W)).reshape(-1) - Y) / unit <= TAU))
        r[kind] = dict(open={s: op[s] for s in range(20)}, closed=cl, gap={s: op[s] - cl[s] for s in range(20)})
    res[ds] = r
    print("%-17s gap median: MSE %.3f  TA %.3f  CLT %.3f | open MSE %.3f CLT %.3f" % (
        ds, *[np.median(list(r[k]["gap"].values())) for k in ("mse", "ta", "clt")],
        np.median(list(r["mse"]["open"].values())), np.median(list(r["clt"]["open"].values()))), flush=True)

def pv(x, alt): x = np.asarray(x); return 1.0 if np.all(x == 0) else float(wilcoxon(x, alternative=alt).pvalue)
out = {}
for k in ("clt", "ta"):
    d = [np.array([res[ds]["mse"]["gap"][s] - res[ds][k]["gap"][s] for s in range(20)]) for ds in DS]
    smaller = holm(np.array([pv(x, "greater") for x in d])); larger = holm(np.array([pv(x, "less") for x in d]))
    red = [float(np.median(x)) for x in d]
    gain = [float(np.median([res[ds][k]["closed"][s] - res[ds]["mse"]["closed"][s] for s in range(20)])) for ds in DS]
    rho_s = float(spearmanr(red, gain).correlation)
    out[k] = dict(n_smaller=int(smaller.sum()), n_larger=int(larger.sum()), spearman=rho_s,
                  per_stream={ds: dict(gap_reduction=red[i], closed_gain=gain[i], smaller=bool(smaller[i]), larger=bool(larger[i])) for i, ds in enumerate(DS)})
    print("%s vs MSE: gap significantly smaller on %d/11, larger on %d/11; Spearman(gap reduction, closed-loop gain) = %.2f" % (k.upper(), smaller.sum(), larger.sum(), rho_s))
H1 = out["clt"]["n_smaller"] >= 6 and out["clt"]["n_larger"] == 0; H2 = out["clt"]["spearman"] > 0.5
print("H1:", "supported" if H1 else "not supported", "| H2:", "supported" if H2 else "not supported")
json.dump(dict(per_stream=res, summary=out, H1=H1, H2=H2), open(os.path.join(HERE, "shift_summary.json"), "w"), indent=1)
