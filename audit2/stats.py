"""Tables 2 (learned columns) and 3 from learned_<ds>.json + audit2/cheap.json."""
import json, numpy as np
import os; WORKDIR = os.path.abspath(os.environ.get("WORKDIR", "."))   # the directory the runs were made in (see README)
from scipy.stats import wilcoxon

DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
      "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
MAC = {"lightweight": 208, "cnn_lstm": 99488}
ER, EOP = 3.8e-4, 1e-9
cheap = json.load(open(f"{WORKDIR}/audit2/cheap.json"))
BONF = 0.05 / 22
print(f"Bonferroni threshold {BONF:.5f}")

def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.1e}"

rows = []
counts = {m: {"saves": 0, "costs": 0, "inconcl": 0, "drho_sig_pos": 0, "drho_sig_neg": 0, "drho_ns": 0}
          for m in MAC}
for ds in DS:
    d = json.load(open(f"{WORKDIR}/learned_{ds}.json"))
    ref = cheap[ds]["taus"]["0.1"]["reference"]
    rho_ref, mac_ref = ref["test"], ref["mac"]
    # sanity: dataset-level fields
    assert d["T"] == cheap[ds]["T"] and d["split"] == [cheap[ds]["tr"], cheap[ds]["va"]], ds
    assert abs(d["unit"] - cheap[ds]["unit"]) < 1e-9 * max(1, d["unit"]), ds
    for m in MAC:
        ents = [e for e in d["entries"] if e["predictor"] == m]
        seeds = sorted(e["seed"] for e in ents)
        assert seeds == list(range(20)), (ds, m, seeds)
        rho = np.array([e["test_rho"] for e in sorted(ents, key=lambda e: e["seed"])])
        dd = rho - rho_ref
        g = ER * dd - (MAC[m] - mac_ref) * EOP
        p_d = wilcoxon(dd, alternative="two-sided").pvalue if np.any(dd != 0) else 1.0
        p_g = wilcoxon(g, alternative="two-sided").pvalue if np.any(g != 0) else 1.0
        sig_g = p_g < BONF
        verdict = ("saves" if g.mean() > 0 else "costs") if sig_g else "inconclusive"
        counts[m][{"saves": "saves", "costs": "costs", "inconclusive": "inconcl"}[verdict]] += 1
        if p_d < BONF:
            counts[m]["drho_sig_pos" if dd.mean() > 0 else "drho_sig_neg"] += 1
        else:
            counts[m]["drho_ns"] += 1
        num = (MAC[m] - mac_ref) * EOP
        if dd.mean() > 0:
            er_star = num / dd.mean()
            if np.all(dd > 0):
                rng = (num / dd.max(), num / dd.min())
                rng_s = f"[{rng[0]:.3g}-{rng[1]:.3g}]"
            else:
                rng_s = f"(no finite range; {int((dd<=0).sum())} seeds with d<=0)"
            er_s = f"{er_star:.3g} {rng_s}"
        else:
            er_s = "never"
        rows.append((ds, m, rho.mean(), rho.std(), dd.mean(), p_d, g.mean() * 1e6, p_g, verdict, er_s))
        print(f"{ds:17s} {m:11s} rho={rho.mean():.3f}±{rho.std():.3f} (ddof1 {rho.std(ddof=1):.3f}) "
              f"ref={ref['arm'][:5]} {rho_ref:.4f} mac_ref={mac_ref:<3d} drho={dd.mean():+.4f} p={fmt_p(p_d):7s} "
              f"g={g.mean()*1e6:+7.1f}uJ p={fmt_p(p_g):7s} {verdict:12s} e_r*={er_s}")
print()
for m in MAC:
    print(m, counts[m])
json.dump(rows, open(f"{WORKDIR}/audit2/table3.json", "w"), indent=1)
