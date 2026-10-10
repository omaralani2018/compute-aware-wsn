"""Version 1.4. Paired quantised-suppression energy check (Section 6.6). Inside run/, next to
ms_numbers.json, learned_*.json, qcl_*.json and the three m0_measure-*.json:
    python3 quant_energy_check.py            -> quant_energy_check.json

For every closed-loop run of quant_closed_loop.py (qcl_<stream>_<model>.json; coverage: CNN-LSTM
intel_time 20 seeds, machine_temp 20, twitter_aapl 20, taxi 15, exchange_cpc 1; lightweight intel_time 20),
the energy verdict of platform_table.py at the anchor e_r is computed twice on the same seeds: with the
floating-point suppression and with the 16-bit fixed-point suppression. Wilcoxon over the checked seeds
(Bonferroni 0.05/22) where there are at least six seeds; with fewer (exchange_cpc), the sign of the mean.
Only these runs are checked with quantised forecasts; every other verdict of Table 6 uses floating-point
suppression. INT-8 is not tested."""
import json, numpy as np
import platform_energy as pe
import platform_table as pt

CFG = [("intel_time", "cnn_lstm"), ("intel_time", "lightweight"), ("machine_temp", "cnn_lstm"),
       ("taxi", "cnn_lstm"), ("twitter_aapl", "cnn_lstm"), ("exchange_cpc", "cnn_lstm")]
PLATFORMS = ["GAP-8", "STM32L4", "STM32H7", "Cortex-M0"]
out, all_d = {}, []
for ds, m in CFG:
    q = json.load(open(f"qcl_{ds}_{m}.json"))
    ref, rr, rmac, rounds, v = pe.stream(ds); rk = pt.REFNAME(ref)
    assert all(abs(v[m][r["seed"]] - r["float"]) < 1e-12 for r in q), (ds, m)   # pairing with learned_*.json
    fl = np.array([r["float"] for r in q]); fx = np.array([r["fixed"] for r in q]); all_d += list(fx - fl)
    res = {}
    for P in PLATFORMS:
        same, vf_all, vq_all = True, [], []
        for lab, E, bits in pt.cases(P):
            Em = E(m) + (pe.PARAMS[m] - pe.REF_PARAMS[rk]) * bits * pe.E_ELEC / pt.H; Er = E(rk)
            gf = pe.E_R * (fl - rr) - (Em - Er); gq = pe.E_R * (fx - rr) - (Em - Er)
            if len(q) >= 6: vf, vq = pe.verdict(gf), pe.verdict(gq)
            else: vf, vq = ("+" if gf.mean() > 0 else "-"), ("+" if gq.mean() > 0 else "-")
            same &= (vf == vq); vf_all.append(vf); vq_all.append(vq)
        res[P] = dict(unchanged=bool(same), float=sorted(set(vf_all)), fixed=sorted(set(vq_all)))
    out[f"{ds}|{m}"] = dict(seeds=[r["seed"] for r in q], diff_mean=float((fx - fl).mean()),
                            diff_min=float((fx - fl).min()), diff_max=float((fx - fl).max()), verdicts=res)
    print(f"{ds:13s} {m:11s} n={len(q):2d} " + " ".join(f"{P}:{'same' if res[P]['unchanged'] else 'CHANGED'}{res[P]['fixed']}" for P in PLATFORMS))
d = np.array(all_d)
out["summary"] = dict(runs=int(len(d)), mean_abs_diff=float(np.abs(d).mean()), max_abs_diff=float(np.abs(d).max()),
                      verdicts_unchanged=all(c["verdicts"][P]["unchanged"] for k, c in out.items() if k != "summary" for P in PLATFORMS))
print(f"{len(d)} runs: mean |diff| {np.abs(d).mean():.4f}, max |diff| {np.abs(d).max():.4f}; every verdict unchanged: {out['summary']['verdicts_unchanged']}")
json.dump(out, open("quant_energy_check.json", "w"), indent=1)
