"""Pre-registered decision analysis for PREREG15 (written before any radio run).
CAPS on 11 streams with each platform's measured forecast energies (as in Section 6.14) and, as the radio cost per
report, the energy of one report measured on that node type's own radio (primary: S_BC; secondary: S_UC), instead of
the modelled anchor. Platforms: M3 software float and M3 integer (IoT-LAB M3 radio), nRF52840 (its radio).
Usage: python3 radio_decisions.py <radio_m3_summary.json> <radio_nrf_summary.json>"""
import os, sys, json, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the caps/ folder
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import frozen_benchmark as fb
ES = fb.E_SENSE
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
ACCQ = json.load(open(os.path.join(SC, "iotlab/fixed/accuracy_q.json")))
R = {"m3": json.load(open(sys.argv[1])), "nrf": json.load(open(sys.argv[2]))}
PLAT = {"m3_float": ("iotlab/iotlab_m3_summary.json", False, "m3"),
        "m3_int": ("iotlab/fixed/fixed_m3_21_accuracy_q_summary.json", True, "m3"),
        "nrf": ("iotlab/iotlab_nrf_summary.json", False, "nrf")}
M = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
out = {}
for variant in ("S_BC", "S_UC"):
    for name, (f, integer, radio) in PLAT.items():
        ER = R[radio]["variants"][variant]["E_inc_J"]
        pf = json.load(open(os.path.join(SC, f)))["per_forecast"]
        E = {M.get(k, k): v["E_inc_J"] for k, v in pf.items()}
        tot = {"ref": 0, "lw": 0, "cnn": 0, "caps": 0, "hind": 0}; picks = {"ref": 0, "lightweight": 0, "cnn_lstm": 0}
        n_pos = {"lightweight": 0, "cnn_lstm": 0}; n_scr = {"lightweight": 0, "cnn_lstm": 0}; wrong = []; per = {}
        for ds, d in MS.items():
            bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
            ea = {"persistence": E["persistence"], "expsm": E["expsm"],
                  "linear_extrap": E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])}
            sel = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], arms.index(a)))
            vR, tR, eR = bt["val"][sel], bt["test"][sel]["rho"], ea[sel]
            L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
            lw = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "lightweight"}
            if integer: lw = {int(s): (v["val_q"], v["test_q"]) for s, v in ACCQ[ds]["seeds"].items()}
            cn = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "cnn_lstm"}
            scr = {k: (E[k] - eR) >= (1 - vR) * ER for k in ("lightweight", "cnn_lstm")}
            en = lambda rho, e: ES + e + (1 - rho) * ER
            row = dict(reference=sel)
            for k, Dk in (("lightweight", lw), ("cnn_lstm", cn)):
                net = [(v[1] - tR) * ER - (E[k] - eR) for v in Dk.values()]
                m = st.median(net); n_pos[k] += m > 0; n_scr[k] += scr[k]
                if scr[k] and m > 0: wrong.append((ds, k))
                row[k] = dict(median_net_uJ=m * 1e6, seeds_positive=sum(x > 0 for x in net), screened=bool(scr[k]))
            caps_e = []
            for s in sorted(lw):
                opts = [("ref", 0.0, en(tR, eR))]
                for k, Dk in (("lightweight", lw), ("cnn_lstm", cn)):
                    if not scr[k] and s in Dk: opts.append((k, (Dk[s][0] - vR) * ER - (E[k] - eR), en(Dk[s][1], E[k])))
                b = max(opts, key=lambda o: o[1]); picks[b[0]] += 1; caps_e.append(b[2])
            e_ref = en(tR, eR); e_lw = st.median(en(v[1], E["lightweight"]) for v in lw.values())
            e_cnn = st.median(en(v[1], E["cnn_lstm"]) for v in cn.values())
            for k, v in (("ref", e_ref), ("lw", e_lw), ("cnn", e_cnn), ("caps", st.median(caps_e)), ("hind", min(e_ref, e_lw, e_cnn))):
                tot[k] += v
            per[ds] = row
        r = {k: 100 * (tot[k] / tot["ref"] - 1) for k in tot}
        res = dict(er_J=ER, positive=n_pos, screened=n_scr, false_exclusions=wrong, picks=picks,
                   caps_vs_ref_pct=r["caps"], hindsight_vs_ref_pct=r["hind"], caps_above_hindsight_pct=100 * (tot["caps"] / tot["hind"] - 1),
                   always_lw_pct=r["lw"], always_cnn_pct=r["cnn"], streams=per,
                   lightweight_positive_streams=[ds for ds, w in per.items() if w["lightweight"]["median_net_uJ"] > 0])
        out[f"{variant}/{name}"] = res
        print("%-5s %-8s er %.3e J | lightweight positive %2d/11 %s | screened lw %2d CNN %2d | false excl. %s | CAPS %+5.2f%% vs ref, %.2f%% above hindsight | picks %s" % (
            variant, name, ER, n_pos["lightweight"], res["lightweight_positive_streams"], n_scr["lightweight"], n_scr["cnn_lstm"],
            wrong or "none", r["caps"], res["caps_above_hindsight_pct"], picks))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "radio_decisions.json"), "w"), indent=1, default=float)
