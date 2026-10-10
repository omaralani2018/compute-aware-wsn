"""Descriptive per-mote and per-block results on the published Intel preparation (V0, test span, tau =
0.10 degC), from the saved per-reading records (v0_cells.py). For each learned model: the suppression
difference from the validation-selected reference (persistence), mean over the 20 seeds, per mote and in
three equal-duration chronological blocks of the test span (a partition of the existing predictions;
predictor state continues across block boundaries), and the net saving at 3.8e-4 J and 1 nJ per MAC.
No significance is claimed: these describe where the effect sits, not uncertainty across deployments.
Writes intel_mote_blocks.json."""
import json, numpy as np
E_R, E_OP = 3.8e-4, 1e-9; MAC = {"lightweight": 208, "cnn_lstm": 99488}
I = np.load("cells/v0_index.npz"); closed = ~I["warmup"]; motes = sorted(set(I["mote"].tolist()))
t = I["bin"]; lo, hi = t[closed].min(), t[closed].max() + 1; edges = np.linspace(lo, hi, 4)
block = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 2)
ref = np.load("cells/v0_persistence.npz")["sent"]
def rho(s, mk): return 1 - s[mk].sum() / mk.sum()
out = {"blocks_utc_edges_s": edges.tolist(), "models": {}}
for m in MAC:
    S = [np.load(f"cells/v0_{m}_s{s}.npz")["sent"] for s in range(20)]
    pm = []
    for mo in motes:
        mk = closed & (I["mote"] == mo)
        d = np.array([rho(s, mk) for s in S]) - rho(ref, mk)
        pm.append(dict(mote=int(mo), steps=int(mk.sum()), delta_mean=float(d.mean()), delta_min=float(d.min()), delta_max=float(d.max()),
                       seeds_above_reference=int((d > 0).sum()), net_saving_uJ=float((E_R * d.mean() - MAC[m] * E_OP) * 1e6)))
    pb = []
    for b in range(3):
        mk = closed & (block == b)
        d = np.array([rho(s, mk) for s in S]) - rho(ref, mk)
        pb.append(dict(block=b + 1, steps=int(mk.sum()), reference_rho=float(rho(ref, mk)), delta_mean=float(d.mean()),
                       seeds_above_reference=int((d > 0).sum()), net_saving_uJ=float((E_R * d.mean() - MAC[m] * E_OP) * 1e6)))
    dm = np.array([x["delta_mean"] for x in pm])
    out["models"][m] = dict(per_mote=pm, per_block=pb, motes_with_mean_delta_above_0=int((dm > 0).sum()), n_motes=len(dm),
                            delta_median=float(np.median(dm)), delta_iqr=[float(np.percentile(dm, 25)), float(np.percentile(dm, 75))],
                            motes_with_positive_mean_net_saving=int(sum(x["net_saving_uJ"] > 0 for x in pm)))
json.dump(out, open("intel_mote_blocks.json", "w"), indent=1)
for m, x in out["models"].items():
    print(m, "motes with mean delta > 0:", x["motes_with_mean_delta_above_0"], "/", x["n_motes"], " median", round(x["delta_median"], 4), "IQR", [round(v, 4) for v in x["delta_iqr"]],
          " motes with positive net saving:", x["motes_with_positive_mean_net_saving"])
    print("   blocks", [(b["block"], round(b["reference_rho"], 4), round(b["delta_mean"], 4), b["seeds_above_reference"], round(b["net_saving_uJ"], 2)) for b in x["per_block"]])
