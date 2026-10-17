"""Record for Section 5.2 and the supplement to Section 5.2 (Figure S3 note): against the low-cost arm with the lowest
test-span energy instead of the validation-selected reference, on which streams the arm differs, whether any mean net
saving changes sign, and the lightweight model's saving on exchange_cpc with its Wilcoxon p (Bonferroni over 22).
Reads only shipped outputs; run from caps/ after caps/setup.sh: python3 records/lowest_energy_arm.py"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(SC, "stage17/run"); sys.path.insert(0, RUN)
import frozen_benchmark as fb
ER, EOP = fb.E_R, fb.E_OP
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]
out = {"differs": [], "sign_changes": [], "streams": {}}
for ds, d in MS.items():
    bt = d["by_tau"]["0.1"]
    macs = {"persistence": 0, "expsm": 12, "linear_extrap": bt["K"]}
    e = {a: macs[a] * EOP + (1 - bt["test"][a]["rho"]) * ER for a in macs}
    low = min(e, key=e.get); ref = bt["selected"]
    if low != ref: out["differs"].append(ds)
    L = json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"]
    rec = {"reference": ref, "lowest_test_energy_arm": low}
    for m in ("lightweight", "cnn_lstm"):
        rho = np.array([x["test_rho"] for x in L if x["predictor"] == m])
        em = fb.MACS[m] * EOP + (1 - rho) * ER
        g_ref, g_low = e[ref] - em, e[low] - em
        rec[m] = {"mean_net_vs_reference_uJ": float(g_ref.mean() * 1e6), "mean_net_vs_lowest_uJ": float(g_low.mean() * 1e6),
                  "p_vs_lowest": float(wilcoxon(g_low).pvalue) if np.any(g_low != 0) else 1.0}
        if np.sign(g_ref.mean()) != np.sign(g_low.mean()): out["sign_changes"].append(f"{ds}/{m}")
    out["streams"][ds] = rec
out["bonferroni_threshold"] = 0.05 / 22
json.dump(out, open(os.path.join(SC, "records/lowest_energy_arm.json"), "w"), indent=1)
x = out["streams"]["exchange_cpc"]["lightweight"]
print("differs on", len(out["differs"]), out["differs"], "| sign changes", out["sign_changes"],
      "| exchange_cpc lightweight %+.1f uJ, p = %.4f (threshold %.4f)" % (x["mean_net_vs_lowest_uJ"], x["p_vs_lowest"], out["bonferroni_threshold"]))
