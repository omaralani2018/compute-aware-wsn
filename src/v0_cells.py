"""Per-reading test records of the published Intel preparation (V0), for the comparison on common
observed timestamps: every column of the test span (with its 12 warm-up rows) is one segment, scored
with segeval.py, which reproduces the main-study evaluator (test_segeval.py). The low-cost arms use the
published alpha and K; the learned models use the shipped weights of all 20 seeds with the
standardisation of run_learned.py. Nothing is trained. Writes cells/v0_*.npz and v0_check.json."""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, frozen_benchmark as fb, fastfwd, segeval as se
LB = fb.LOOK_BACK
A = np.load("intel_time.npy"); T = len(A); tr, va = int(.5 * T), int(.7 * T); TEST = A[va - LB:]
rows = pd.to_datetime(pd.read_csv("intel_time_rows.csv")["bin_start"])
secs = (rows - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()[va - LB:]
meta = json.load(open("intel_time_meta.json")); motes = meta["motes"]
# observed (not interpolated) cells: compare with the raw bin means used by intel_variants.py
import intel_variants as iv
p, pt = iv.load_bins(); P = p.reindex(rows).to_numpy()[va - LB:]
observed = ~np.isnan(P)
segs = [TEST[:, j] for j in range(TEST.shape[1])]
np.savez_compressed("cells/v0_index.npz", mote=np.concatenate([np.full(len(TEST), m) for m in motes]),
                    bin=np.concatenate([secs for _ in motes]), observed=np.concatenate([observed[:, j] for j in range(len(motes))]),
                    warmup=np.concatenate([np.arange(len(TEST)) < LB for _ in motes]))
def save(name, f):
    r = se.seg_dps(segs, f, 0.1, record=True)
    np.savez_compressed(f"cells/v0_{name}.npz", sent=np.concatenate(r["sent"]), err=np.concatenate(r["err"]))
    return r["rho"]
M = json.load(open("ms_numbers.json"))["datasets"]["intel_time"]["by_tau"]["0.1"]
check = {}
for n, f in {"persistence": None, "expsm": fb.make_expsm(M["alpha"]), "linear_extrap": fb.make_ar(M["K"])}.items():
    check[n] = (save(n, f), M["test"][n]["rho"])
rng = np.random.default_rng(0); Xtr, Ytr = fb.windows(A, A, LB, tr, rng)
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
L = {(r["predictor"], r["seed"]): r["test_rho"] for r in json.load(open("learned_intel_time.json"))["entries"]}
for kind in ("lightweight", "cnn_lstm"):
    for sd in range(20):
        m = fb.build(kind, sd); w = np.load(f"weights/intel_time_{kind}_s{sd}.npz"); m.set_weights([w[k] for k in w.files])
        Pm = fastfwd.extract(m, kind)
        rho = save(f"{kind}_s{sd}", lambda W, Pm=Pm: fastfwd.forward(Pm, (np.asarray(W, float) - mx) / sx) * sy + my)
        check[f"{kind}_s{sd}"] = (rho, L[(kind, sd)])
bad = {k: v for k, v in check.items() if abs(v[0] - v[1]) > 1e-12}
json.dump(dict(check=check, mismatches=bad), open("v0_check.json", "w"), indent=1)
print("mismatches:", bad)
