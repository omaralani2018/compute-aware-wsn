"""Pre-registered analysis for PREREG3.txt. Written before any TA result was inspected."""
import os, json, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]
ER, EOP, ALPHA = 3.8e-4, 1e-9, 0.05
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
      "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]

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

res = {}
for ds in DS:
    ta = {r["seed"]: r for r in json.load(open(os.path.join(HERE, f"ta_{ds}.json")))["entries"]}
    mse = {r["seed"]: r for r in json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"]
           if r["predictor"] == "lightweight"}
    seeds = sorted(set(ta) & set(mse)); assert len(seeds) == 20, (ds, len(seeds))
    bt = MS[ds]["by_tau"]["0.1"]; rR = bt["test"][bt["selected"]]["rho"]; dM = (208 - bt["selected_macs"]) * EOP
    t = np.array([ta[s]["test_rho"] for s in seeds]); m = np.array([mse[s]["test_rho"] for s in seeds])
    netT = (t - rR) * ER - dM; netM = (m - rR) * ER - dM
    res[ds] = dict(rho_R=rR, ta_med=float(np.median(t)), mse_med=float(np.median(m)),
                   d_med=float(np.median(t - m)), p_h1=pv(t - m, "greater"), p_h1_rev=pv(t - m, "less"),
                   p_h2_ta=pv(netT, "greater"), p_h2_mse=pv(netM, "greater"),
                   verified=all(ta[s]["receiver_verified"] for s in seeds),
                   c_counts={c: sum(ta[s]["c"] == c for s in seeds) for c in (0.25, 0.5, 1.0)})
h1 = holm([res[d]["p_h1"] for d in DS]); h1r = holm([res[d]["p_h1_rev"] for d in DS])
h2t = holm([res[d]["p_h2_ta"] for d in DS]); h2m = holm([res[d]["p_h2_mse"] for d in DS])
for i, d in enumerate(DS):
    res[d].update(H1=bool(h1[i]), H1_rev=bool(h1r[i]), H2_ta=bool(h2t[i]), H2_mse=bool(h2m[i]))
    r = res[d]
    print("%-17s rhoR=%.3f MSE=%.3f TA=%.3f d=%+.3f  H1=%-5s rev=%-5s  beatsR: MSE=%-5s TA=%-5s c=%s verified=%s" % (
        d, r["rho_R"], r["mse_med"], r["ta_med"], r["d_med"], r["H1"], r["H1_rev"], r["H2_mse"], r["H2_ta"], r["c_counts"], r["verified"]))
n1, nrev = int(h1.sum()), int(h1r.sum()); n2t, n2m = int(h2t.sum()), int(h2m.sum())
go = (n1 >= 4 and nrev == 0) and (n2t - n2m >= 2)
print(f"H1 significant: {n1}/11, reversed: {nrev}; learned beats R: MSE {n2m}, TA {n2t}  ->  {'GO' if go else 'NO-GO'}")
json.dump({"per_stream": res, "n_h1": n1, "n_rev": nrev, "n_beatR_mse": n2m, "n_beatR_ta": n2t, "GO": go},
          open(os.path.join(HERE, "ta_summary.json"), "w"), indent=1)
