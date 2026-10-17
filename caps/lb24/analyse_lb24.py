"""Pre-registered analysis for PREREG9.txt (look-back 24). Written before any lb24 result was inspected.
R1: seed-median test suppression of the reference and of the lightweight model, look-back 12 vs 24.
R2: energy verdict of the lightweight model against the reference, one-sided Wilcoxon over seeds in both
    directions, Holm across the 11 streams per direction; verdict = save / cost / ns. Flip = save <-> cost.
Conclusion rule (fixed in PREREG9): insensitive to the look-back if no stream flips."""
import os, json, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]
ER, EOP, ALPHA = 3.8e-4, 1e-9, 0.05
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
      "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
M_LW = {12: 208, 24: 400}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    for i, k in enumerate(o):
        if p[k] <= ALPHA / (m - i): rej[k] = True
        else: break
    return rej

def pv(x, alt):
    x = np.asarray(x, float)
    if np.all(x == 0): return 1.0
    return float(wilcoxon(x, alternative=alt, zero_method="wilcox").pvalue)

def setting(lb, ds):
    if lb == 12:
        bt = MS[ds]["by_tau"]["0.1"]; sel, mR, rR = bt["selected"], bt["selected_macs"], bt["test"][bt["selected"]]["rho"]
        E = [r for r in json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"] if r["predictor"] == "lightweight"]
    else:
        ch = json.load(open(os.path.join(HERE, f"cheap_{ds}.json"))); sel, mR, rR = ch["selected"], ch["selected_macs"], ch["test"][ch["selected"]]
        E = [r for r in json.load(open(os.path.join(HERE, f"learned_{ds}.json")))["entries"] if r["predictor"] == "lightweight"]
    E = sorted(E, key=lambda r: r["seed"]); assert len(E) == 20 and len({r["seed"] for r in E}) == 20, (lb, ds, len(E))
    rho = np.array([r["test_rho"] for r in E]); net = (rho - rR) * ER - (M_LW[lb] - mR) * EOP
    return dict(ref=sel, ref_macs=mR, rho_ref=rR, rho_lw_med=float(np.median(rho)), net_med=float(np.median(net)),
                p_save=pv(net, "greater"), p_cost=pv(net, "less"), verified=all(r["receiver_verified"] for r in E))

res = {lb: {ds: setting(lb, ds) for ds in DS} for lb in (12, 24)}
for lb in (12, 24):
    s = holm([res[lb][d]["p_save"] for d in DS]); c = holm([res[lb][d]["p_cost"] for d in DS])
    for i, d in enumerate(DS): res[lb][d]["verdict"] = "save" if s[i] else "cost" if c[i] else "ns"
flips = [d for d in DS if {res[12][d]["verdict"], res[24][d]["verdict"]} == {"save", "cost"}]
changes = [d for d in DS if res[12][d]["verdict"] != res[24][d]["verdict"]]
for d in DS:
    a, b = res[12][d], res[24][d]
    print("%-17s ref12=%-13s rhoR %.3f lw %.3f %-4s | ref24=%-13s rhoR %.3f lw %.3f %-4s | verified %s" % (
        d, a["ref"], a["rho_ref"], a["rho_lw_med"], a["verdict"], b["ref"], b["rho_ref"], b["rho_lw_med"], b["verdict"], b["verified"]))
cnt = {lb: {v: sum(res[lb][d]["verdict"] == v for d in DS) for v in ("save", "cost", "ns")} for lb in (12, 24)}
print("look-back 12:", cnt[12], "| look-back 24:", cnt[24])
print("verdict changes:", changes, "| save<->cost flips:", flips)
print("CONCLUSION:", "insensitive to the look-back" if not flips else "SENSITIVE to the look-back")
json.dump(dict(per_stream=res, counts=cnt, changes=changes, flips=flips, insensitive=not flips),
          open(os.path.join(HERE, "lb24_summary.json"), "w"), indent=1, default=float)
