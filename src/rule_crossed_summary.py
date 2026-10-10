"""Version 1.4. Reads the eleven rule_crossed_<stream>.json and writes
rule_crossed_summary.csv (one row per stream, scheme, parameter set and rule
variant) and prints the numbers quoted in Section 6.8. Inside run/:
    python3 rule_crossed_summary.py"""
import json, csv
DS = ['intel_time', 'taxi', 'traffic', 'cpu', 'machine_temp', 'grok_asg_anomaly', 'rds_cpu_util',
      'iio_networkin', 'twitter_aapl', 'exchange_cpc', 'occupancy']
RULES = ("cons_lt", "cons_le", "single_le", "single_lt")
rows, po, pa, skip, ties, worst = [], [], [], [], 0, 0.0
for ds in DS:
    c = json.load(open(f"rule_crossed_{ds}.json"))["cells"]
    for sc in ("LMS", "AMDR"):
        for src in ("authors", "ours"):
            x = c[f"{sc}_sel_{src}"]
            for r in RULES:
                y = x[r]; ties += y["ties_at_tau"]; worst = max(worst, y["max_err_over_tau_unit"])
                rows.append([ds, sc, "released" if src == "authors" else "single-error",
                             json.dumps(x["params"], sort_keys=True), x["macs"], r, y["rho"], y["n_tx"],
                             y["n_steps"], y["mean_abs_err"], y["max_err_over_tau_unit"], y["ties_at_tau"]])
        o, a = c[f"{sc}_sel_ours"], c[f"{sc}_sel_authors"]
        po.append(o["single_le"]["rho"] - o["cons_le"]["rho"])
        if a["single_le"]["rho"] == 0 and a["cons_le"]["rho"] == 0: skip.append(f"{sc}/{ds}")
        else: pa.append(a["single_le"]["rho"] - a["cons_le"]["rho"])
with open("rule_crossed_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stream", "scheme", "parameters_selected_under", "parameters", "macs", "rule",
                "rho", "n_tx", "n_steps", "mean_abs_err", "max_err_over_tau_unit", "errors_equal_to_tau"])
    w.writerows(rows)
print(f"runs {len(rows)}, errors equal to tau {ties}, largest sink error {worst:.6f} tau (bound: 1)")
print(f"single-error parameters: released rule lower on {sum(d > 0 for d in po)} of {len(po)} pairs, "
      f"by {min(po):.2f}-{max(po):.2f}")
print(f"released-rule parameters: single-error rule higher on {sum(d > 0 for d in pa)} of {len(pa)} pairs, "
      f"by {min(pa):.2f}-{max(pa):.2f}; no suppression under either rule: {', '.join(skip)}")
