"""PREREG16 (sha256 77b812f5...): compute-blind selection vs CAPS, always-reference and hindsight.
Configurations A (1 nJ/MAC, radio sweep) and B (platforms at the anchor) on the 11 development streams."""
import os, sys, json, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import frozen_benchmark as fb
ES = fb.E_SENSE
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
ACCQ = json.load(open(os.path.join(SC, "iotlab/fixed/accuracy_q.json")))
def load(f): return json.load(open(os.path.join(SC, f)))
MAC = {"lightweight": 208, "cnn_lstm": 99488}
def energies(f):
    if f is None: return None
    pf = load(f)["per_forecast"]; key = "E_inc_J" if "E_inc_J" in next(iter(pf.values())) else None
    m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
    return {m.get(k, k): (v[key] if key else v["E_uJ"] * 1e-6) for k, v in pf.items()}
def run(E, ER, integer):
    tot = {"caps": 0, "blind": 0, "ref": 0, "hind": 0}; picks = {"caps": {}, "blind": {}}; per = {}
    for ds, d in MS.items():
        bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
        if E is None:
            ea = {"persistence": 0.0, "expsm": 12e-9, "linear_extrap": bt["K"] * 1e-9}; El = {k: MAC[k] * 1e-9 for k in MAC}
            sel_c = bt["selected"]; ea[sel_c] = bt["selected_macs"] * 1e-9
        else:
            ea = {"persistence": E["persistence"], "expsm": E["expsm"], "linear_extrap": E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])}
            El = {k: E[k] for k in MAC}
            sel_c = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], arms.index(a)))
        sel_b = bt["selected"]  # highest validation suppression (paper's reference rule)
        L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
        lw = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "lightweight"}
        if integer: lw = {int(s): (v["val_q"], v["test_q"]) for s, v in ACCQ[ds]["seeds"].items()}
        cn = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "cnn_lstm"}
        en = lambda rho, e: ES + e + (1 - rho) * ER
        vR, tR, eR = bt["val"][sel_c], bt["test"][sel_c]["rho"], ea[sel_c]
        scr = {k: (El[k] - eR) >= (1 - vR) * ER for k in MAC}
        vB, tB, eB = bt["val"][sel_b], bt["test"][sel_b]["rho"], ea[sel_b]
        ce, be = [], []
        for s in sorted(lw):
            opts = [("ref", 0.0, en(tR, eR))]
            for k, D in (("lightweight", lw), ("cnn_lstm", cn)):
                if not scr[k] and s in D: opts.append((k, (D[s][0] - vR) * ER - (El[k] - eR), en(D[s][1], El[k])))
            b = max(opts, key=lambda o: o[1]); picks["caps"][b[0]] = picks["caps"].get(b[0], 0) + 1; ce.append(b[2])
            bo = [("ref", vB, 0, en(tB, eB))] + [(k, D[s][0], MAC[k], en(D[s][1], El[k])) for k, D in (("lightweight", lw), ("cnn_lstm", cn)) if s in D]
            bb = max(bo, key=lambda o: (o[1], -o[2])); picks["blind"][bb[0]] = picks["blind"].get(bb[0], 0) + 1; be.append(bb[3])
        e_ref = en(tR, eR); e_lw = st.median(en(v[1], El["lightweight"]) for v in lw.values()); e_cn = st.median(en(v[1], El["cnn_lstm"]) for v in cn.values())
        row = {"caps": st.median(ce), "blind": st.median(be), "ref": e_ref, "hind": min(e_ref, e_lw, e_cn)}
        for k in tot: tot[k] += row[k]
        per[ds] = row
    return tot, picks, per
OUT = {}
for ER in (1e-4, 3.75e-4, 3.8e-4, 1e-3, 3e-3):
    tot, picks, per = run(None, ER, False); OUT[f"A_1nJ_er{ER:g}"] = {"tot": tot, "picks": picks, "per": per}
for name, f, integer in (("ATmega128L", "avrsim/avr_summary.json", False), ("M3_float", "iotlab/iotlab_m3_summary.json", False),
                         ("M3_int", "iotlab/fixed/fixed_m3_21_accuracy_q_summary.json", True), ("nRF52840", "iotlab/iotlab_nrf_summary.json", False)):
    tot, picks, per = run(energies(f), 3.8e-4, integer); OUT[f"B_{name}"] = {"tot": tot, "picks": picks, "per": per}
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind_vs_caps.json"), "w"), indent=1)
for k, v in OUT.items():
    t = v["tot"]
    print("%-18s blind vs CAPS %+8.1f%% | CAPS vs ref %+5.1f%% | blind vs ref %+8.1f%% | hindsight vs ref %+5.1f%% | blind picks %s | CAPS picks %s" % (
        k, 100 * (t["blind"] / t["caps"] - 1), 100 * (t["caps"] / t["ref"] - 1), 100 * (t["blind"] / t["ref"] - 1), 100 * (t["hind"] / t["ref"] - 1), v["picks"]["blind"], v["picks"]["caps"]))
