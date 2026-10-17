"""Record for Section 5.1 (integer code): the choice of CAPS on the M3 in integer code with the firmware's single input
scale (accuracy_q_v1_fixed_si20.json) against the per-stream scale used in the paper (accuracy_q.json). Same rule as
capsplat/caps_platforms.py (incremental energy). Run from caps/ after caps/setup.sh: python3 records/caps_single_scale.py"""
import os, sys, json, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import frozen_benchmark as fb
ER, ES = 3.8e-4, fb.E_SENSE
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
def caps(acc_file, summ_file):
    ACCQ = json.load(open(os.path.join(SC, "iotlab/fixed", acc_file)))
    pf = json.load(open(os.path.join(SC, "iotlab/fixed", summ_file)))["per_forecast"]
    m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
    E = {m.get(k, k): v["E_inc_J"] for k, v in pf.items()}
    picks, tot = {}, {"ref": 0.0, "caps": 0.0}
    for ds, d in MS.items():
        bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
        ea = {"persistence": E["persistence"], "expsm": E["expsm"], "linear_extrap": E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])}
        sel = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], arms.index(a)))
        vR, tR, eR = bt["val"][sel], bt["test"][sel]["rho"], ea[sel]
        L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
        lw = {int(s): (v["val_q"], v["test_q"]) for s, v in ACCQ[ds]["seeds"].items()}
        cn = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "cnn_lstm"}
        scr = {k: (E[k] - eR) >= (1 - vR) * ER for k in ("lightweight", "cnn_lstm")}
        en = lambda rho, e: ES + e + (1 - rho) * ER
        ce = []
        for s in sorted(lw):
            opts = [("ref", 0.0, en(tR, eR))]
            for k, D in (("lightweight", lw), ("cnn_lstm", cn)):
                if not scr[k] and s in D: opts.append((k, (D[s][0] - vR) * ER - (E[k] - eR), en(D[s][1], E[k])))
            b = max(opts, key=lambda o: o[1]); picks[f"{ds}/{s}"] = b[0]; ce.append(b[2])
        tot["ref"] += en(tR, eR); tot["caps"] += st.median(ce)
    return picks, 100 * (tot["caps"] / tot["ref"] - 1)
p1, r1 = caps("accuracy_q.json", "fixed_m3_21_accuracy_q_summary.json")
p2, r2 = caps("accuracy_q_v1_fixed_si20.json", "fixed_m3_21_accuracy_q_v1_fixed_si20_summary.json")
out = {"per_stream_scale": {"caps_vs_ref_pct": r1}, "single_scale": {"caps_vs_ref_pct": r2},
       "seed_choices": len(p1), "identical_choices": sum(p1[k] == p2[k] for k in p1)}
json.dump(out, open(os.path.join(SC, "records/caps_single_scale.json"), "w"), indent=1)
print("choices identical %d of %d; CAPS vs reference %+.2f%% (per-stream scale) and %+.2f%% (single scale)" % (out["identical_choices"], len(p1), r1, r2))
