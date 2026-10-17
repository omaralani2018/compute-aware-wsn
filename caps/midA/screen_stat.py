"""PREREG14 B: out-of-sample check of the Section 6.6 screening statistic on the 13 datasets not used in development."""
import os, sys, json, pickle, glob, statistics as st
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import numpy as np
from scipy.stats import spearmanr
import frozen_benchmark as fb
TAU, LB = 0.10, fb.LOOK_BACK

def arr_case(A, u):
    A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = len(A); tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    d = np.abs(np.diff(A[:tr], axis=0)); stat = float((d <= TAU * u).mean())
    rho = fb.dps(A[va - LB:], None, tau=TAU, look_back=LB, train_frac=1e-6, unit=u)["rho"]
    v = A[:tr]; fast = float(np.mean([np.std(np.diff(v[:, j])) / np.std(v[:, j]) for j in range(v.shape[1])]))
    return stat, rho, fast

def seg_case(segfile, codedir):
    sys.path.insert(0, codedir); import segeval as se
    S = pickle.load(open(segfile, "rb"))
    tr = [np.asarray(s["values"], float) for s in S if s["span"] == "train"]
    d = np.concatenate([np.abs(np.diff(x)) for x in tr if len(x) > 1]); stat = float((d <= TAU).mean())
    rho = se.seg_dps([s["values"] for s in S if s["span"] == "test"], None, TAU)["rho"]
    fast = float(np.mean([np.std(np.diff(x)) / np.std(x) for x in tr if len(x) > LB and np.std(x) > 0]))
    sys.path.remove(codedir); del sys.modules["segeval"]
    return stat, rho, fast

res = {}
res["grand_st_bernard"] = seg_case(os.path.join(SC, "stb_exec/stb_segments_obs.pkl"), os.path.join(SC, "stb_exec"))
for dep in ["fishnet", "genepi", "pdg2008", "plaine_morte"]:
    res[dep] = seg_case(os.path.join(SC, f"confB/run_{dep}/stb_segments_obs.pkl"), os.path.join(SC, "confB/code"))
for grp in ("confA", "confC"):
    sys.path.insert(0, os.path.join(SC, grp)); import conf_data as cd
    for ds in cd.FILES:
        A, u = cd.load(ds); res[ds] = arr_case(A, u)
    sys.path.remove(os.path.join(SC, grp)); del sys.modules["conf_data"]
x = np.array([v[0] for v in res.values()]); y = np.array([v[1] for v in res.values()])
r13 = spearmanr(x, y); p1 = r13.pvalue / 2 if r13.correlation > 0 else 1 - r13.pvalue / 2
mae = float(np.mean(np.abs(x - y))); worst = float(np.max(np.abs(x - y)))
for k, (s_, r_, f_) in res.items(): print("%-20s statistic %.3f  persistence test rho %.3f  |diff| %.3f  fastness(train) %.3f" % (k, s_, r_, abs(s_ - r_), f_))
print("13 new datasets: Spearman %.2f (one-sided p %.4f), MAE %.3f, worst %.3f -> %s" % (
    r13.correlation, p1, mae, worst, "CONFIRMED" if (r13.correlation >= 0.6 and p1 < 0.05 and mae <= 0.10) else "NOT CONFIRMED"))
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
xd = [MS[d]["by_tau"]["0.1"]["open_loop_frac_train"] for d in MS]; yd = [MS[d]["by_tau"]["0.1"]["test"]["persistence"]["rho"] for d in MS]
xa, ya = np.r_[x, xd], np.r_[y, yd]
print("pooled 24: Spearman %.2f, MAE %.3f, worst %.3f" % (spearmanr(xa, ya).correlation, np.mean(np.abs(xa - ya)), np.max(np.abs(xa - ya))))
fr = np.array([v[2] for v in res.values()]); print("fastness(train) vs persistence rho, 13 new: Spearman %.2f" % spearmanr(fr, y).correlation)
# descriptive: statistic vs lightweight median suppression difference from the reference
def lw_delta_dev(d):
    bt = MS[d]["by_tau"]["0.1"]; tR = bt["test"][bt["selected"]]["rho"]
    L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{d}.json")))["entries"]
    return st.median(e["test_rho"] for e in L if e["predictor"] == "lightweight") - tR
dl = {d: lw_delta_dev(d) for d in MS}
for grp, key in (("confA/conf_summary.json", "streams"), ("confC/conf_summary.json", "streams"), ("confB/confB_summary.json", "deployments")):
    J = json.load(open(os.path.join(SC, grp)))[key]
    for d, v in J.items():
        if grp.startswith("confB"):
            rows = [json.load(open(f)) for f in glob.glob(os.path.join(SC, f"confB/run_{d}/runs/stb_obs_lightweight_s*.json"))]
            lw = [r["test_rho"] for r in rows if r.get("status") == "ok"]
        else:
            L = json.load(open(os.path.join(SC, grp.split("/")[0], f"learned_{d}.json")))["entries"]; lw = [e["test_rho"] for e in L if e["predictor"] == "lightweight"]
        dl[d] = st.median(lw) - v["rho_ref_test"]
G = json.load(open(os.path.join(SC, "stb_exec/stb_summary.json")))["primary_obs"]
rows = [json.load(open(f)) for f in glob.glob(os.path.join(SC, "stb_exec/runs/stb_obs_lightweight_s*.json"))]
dl["grand_st_bernard"] = st.median(r["test_rho"] for r in rows if r.get("status") == "ok") - G["reference_rho"]
assert abs(res["grand_st_bernard"][1] - G["low_cost_test"]["persistence"]["rho"]) < 1e-9
stat_all = dict(zip(list(res) + list(MS), list(x) + xd))
common = [d for d in stat_all if d in dl]
print("descriptive, %d datasets: Spearman(statistic, lightweight median delta-rho vs reference) = %.2f" % (
    len(common), spearmanr([stat_all[d] for d in common], [dl[d] for d in common]).correlation))
json.dump(dict(new=res, spearman13=r13.correlation, p_one_sided=p1, mae13=mae, worst13=worst, lw_delta=dl), open(os.path.join(HERE, "screen_stat.json"), "w"), indent=1, default=float)
