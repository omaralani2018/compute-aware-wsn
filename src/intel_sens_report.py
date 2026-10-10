"""Intel preprocessing sensitivity: summary of the three preparations.
V0      published matrix (interpolation of up to 8 bins of every interior gap before the split, complete rows)
V_obs   observed readings only, per mote, segments cut at split boundaries and gaps > 300 s
V_short V_obs segments with gaps of up to 300 s (about 9 missing bins, close to V0's 8) filled by linear interpolation in elapsed time
V_short100 the same segments, with only gaps of up to 100 s (at most 2 missing bins) filled; longer gaps skipped as in V_obs
Reports coverage, initialisation, suppression, errors, modelled energy and verdicts at the anchor
(3.8e-4 J, 1 nJ per MAC), rankings, and the comparison on common observed timestamps.
Statistics as in the main study (Wilcoxon over 20 seeds, threshold 0.05/22), with the exact sign-flip
permutation test of the mean reported alongside. Writes intel_sensitivity.json."""
import json, numpy as np
from scipy.stats import wilcoxon
E_R, E_OP, E_S = 3.8e-4, 1e-9, 1e-7
MAC = {"persistence": 0, "expsm": 12, "lightweight": 208, "cnn_lstm": 99488}
THR_MAIN = 0.05 / 22
def p_w(x): return float(wilcoxon(x)[1]) if np.any(x != 0) else 1.0
def p_perm(x):
    x = np.asarray(x, float); n = len(x)
    if not np.any(x != 0): return 1.0
    s = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2.0 - 1.0
    return float(np.mean(np.abs(s @ x) >= abs(x.sum()) * (1 - 1e-12)))
def verdict(x, p): return ("+" if x.mean() > 0 else "-") if p < THR_MAIN else "ns"

def macs(name, K): return K if name == "linear_extrap" else MAC[name]
def load(v):
    if v == "V0":
        M = json.load(open("ms_numbers.json"))["datasets"]["intel_time"]["by_tau"]["0.1"]
        L = json.load(open("learned_intel_time.json"))["entries"]
        cheap = {n: dict(rho=t["rho"], mean_err=t["mean_err"], rmse=t["rmse"], max_err=t["max_err"]) for n, t in M["test"].items()}
        learned = {k: sorted([e for e in L if e["predictor"] == k], key=lambda e: e["seed"]) for k in ("lightweight", "cnn_lstm")}
        for k in learned:
            for e in learned[k]: e.setdefault("test_rho_incl_init", None)
        return dict(alpha=M["alpha"], K=M["K"], selected=M["selected"], cheap=cheap, learned=learned)
    key = {"V_obs": "obs", "V_short": "short", "V_short100": "short100"}[v]
    C = json.load(open(f"sens_{key}_cheap.json"))
    learned = {k: sorted(json.load(open(f"sens_{key}_{k}.json"))["entries"], key=lambda e: e["seed"]) for k in ("lightweight", "cnn_lstm")}
    return dict(alpha=C["alpha"], K=C["K"], selected=C["selected"], cheap=C["test"], learned=learned)

out = {"preparations": {}}
cov = json.load(open("intel_variants_coverage.json"))
for v in ("V0", "V_obs", "V_short", "V_short100"):
    D = load(v); sel = D["selected"]; rr = D["cheap"][sel]["rho"]; rm = macs(sel, D["K"])
    R = dict(alpha=D["alpha"], K=D["K"], reference=sel, reference_rho=rr,
             low_cost={n: {k: t.get(k) for k in ("rho", "rho_incl_init", "mean_err", "rmse", "max_err", "receiver_verified", "n_steps", "n_init_reports")}
                       for n, t in D["cheap"].items()}, learned={})
    for k, E in D["learned"].items():
        assert len(E) == 20 and [e["seed"] for e in E] == list(range(20)), (v, k, len(E))
        r = np.array([e["test_rho"] for e in E]); dd = r - rr; g = E_R * dd - (MAC[k] - rm) * E_OP
        R["learned"][k] = dict(rho_mean=float(r.mean()), rho_sd=float(r.std()),
                               mean_err=float(np.mean([e["test_mean_err"] for e in E])), rmse=float(np.mean([e.get("test_rmse", np.nan) for e in E])),
                               max_err=float(np.max([e["test_max_err"] for e in E])),
                               receiver_all_verified=all(e["receiver_verified"] for e in E),
                               delta_mean=float(dd.mean()), p_delta_wilcoxon=p_w(dd), p_delta_perm=p_perm(dd), verdict_delta=verdict(dd, p_w(dd)),
                               net_saving_uJ=float(g.mean() * 1e6), p_net_wilcoxon=p_w(g), p_net_perm=p_perm(g), verdict_energy=verdict(g, p_w(g)),
                               selected_epochs=[e["selected_epoch"] for e in E])
    arms = {n: R["low_cost"][n]["rho"] for n in R["low_cost"]} | {k: R["learned"][k]["rho_mean"] for k in R["learned"]}
    R["ties_in_suppression"] = sorted({tuple(sorted(n for n in arms if arms[n] == x)) for x in arms.values() if list(arms.values()).count(x) > 1})
    R["rank_by_suppression"] = sorted(arms, key=lambda n: -arms[n])
    energy = {n: E_S + macs(n, D["K"]) * E_OP + (1 - arms[n]) * E_R for n in arms}          # per scored step, anchor, 1 nJ
    R["energy_per_step_uJ"] = {n: energy[n] * 1e6 for n in arms}
    R["rank_by_energy"] = sorted(arms, key=lambda n: energy[n])
    out["preparations"][v] = R
out["coverage"] = dict(V0=cov["V0"], V_obs=cov["variants"]["obs"], V_short=cov["variants"]["short"], V_short100=cov["variants"]["short100"],
                       strict_contiguous_coverage_only=cov["strict_contiguous_coverage_only"])

# common observed timestamps: observed readings scored (not initialisation) in all three preparations
def index(v):
    I = np.load(f"cells/{'v0' if v == 'V0' else v[2:]}_index.npz")
    keep = ~I["warmup"] & (I["observed"] if v == "V0" else ~I["filled"])
    return {(int(m), float(b)): i for i, (m, b, k) in enumerate(zip(I["mote"], I["bin"], keep)) if k}
ix = {v: index(v) for v in ("V0", "V_obs", "V_short", "V_short100")}
common = sorted(set(ix["V0"]) & set(ix["V_obs"]) & set(ix["V_short"]) & set(ix["V_short100"]))
out["common_observed"] = dict(n=len(common), arms={})
pref = {"V0": "v0", "V_obs": "obs", "V_short": "short", "V_short100": "short100"}
for arm in ("persistence", "expsm", "linear_extrap", "lightweight", "cnn_lstm"):
    row = {}
    for v in ("V0", "V_obs", "V_short", "V_short100"):
        idx = np.array([ix[v][c] for c in common])
        files = [f"cells/{pref[v]}_{arm}.npz"] if arm in ("persistence", "expsm", "linear_extrap") else [f"cells/{pref[v]}_{arm}_s{s}.npz" for s in range(20)]
        S = [np.load(f) for f in files]
        sent = np.mean([s["sent"][idx] for s in S], 0); err = np.mean([np.abs(s["err"][idx]) for s in S], 0)
        row[v] = dict(suppression=float(1 - sent.mean()), mae=float(err.mean()), max_err=float(np.max([np.abs(s["err"][idx]).max() for s in S])))
    out["common_observed"]["arms"][arm] = row
# cross-preparation change of each learned model's mean suppression (different seeds' models, unpaired)
from scipy.stats import mannwhitneyu
out["across_preparations"] = {}
for k in ("lightweight", "cnn_lstm"):
    base = np.array([e["test_rho"] for e in load("V0")["learned"][k]])
    out["across_preparations"][k] = {v: dict(change=float(np.mean([e["test_rho"] for e in load(v)["learned"][k]]) - base.mean()),
        p_mann_whitney=float(mannwhitneyu([e["test_rho"] for e in load(v)["learned"][k]], base).pvalue)) for v in ("V_obs", "V_short", "V_short100")}
json.dump(out, open("intel_sensitivity.json", "w"), indent=1)
for v, R in out["preparations"].items():
    print(v, "ref", R["reference"], round(R["reference_rho"], 4), {k: (round(x["rho_mean"], 4), x["verdict_delta"], round(x["net_saving_uJ"], 2), x["verdict_energy"]) for k, x in R["learned"].items()})
    print("   rank rho", [(n, round(arms_, 4)) for n, arms_ in sorted(((n, R["low_cost"][n]["rho"] if n in R["low_cost"] else R["learned"][n]["rho_mean"]) for n in R["rank_by_suppression"]), key=lambda x: -x[1])])
print("common observed", out["common_observed"]["n"])
for a, r in out["common_observed"]["arms"].items(): print(" ", a, {v: (round(x["suppression"], 4), round(x["mae"], 4)) for v, x in r.items()})
