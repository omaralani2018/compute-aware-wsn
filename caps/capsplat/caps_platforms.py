"""CAPS Step 2 per seed with each platform's measured/simulated energies (deterministic application of Algorithm 1,
added after the platform results; no parameter). Energy per node-round = Esense + Ecompute + (1 - rho) * er."""
import os, sys, json, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import frozen_benchmark as fb
ER, ES = 3.8e-4, fb.E_SENSE
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
ACCQ = json.load(open(os.path.join(SC, "iotlab/fixed/accuracy_q.json")))
def load(f): return json.load(open(os.path.join(SC, f)))
PLAT = {
 "anchor":   (None, False),
 "mica2":    ("avrsim/avr_summary.json", False),
 "m3_float": ("iotlab/iotlab_m3_summary.json", False),
 "m3_int":   ("iotlab/fixed/fixed_m3_21_accuracy_q_summary.json", True),
 "nrf":      ("iotlab/iotlab_nrf_summary.json", False),
}
for name, (f, integer) in PLAT.items():
    for gross in ((False, True) if f and "iotlab" in f else (False,)):
        if f is None:
            E = {"persistence": 0.0, "expsm": 12e-9, "lin2": 2e-9, "lin12": 12e-9, "lightweight": 208e-9, "cnn_lstm": 99488e-9}
        else:
            pf = load(f)["per_forecast"]; key = "E_gross_J" if gross else ("E_inc_J" if "E_inc_J" in next(iter(pf.values())) else None)
            m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
            E = {m.get(k, k): (v[key] if key else v["E_uJ"] * 1e-6) for k, v in pf.items()}
        tot = {"ref": 0, "lw": 0, "cnn": 0, "caps": 0, "hind": 0}; picks = {"ref": 0, "lightweight": 0, "cnn_lstm": 0}
        for ds, d in MS.items():
            bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
            ea = {"persistence": E["persistence"], "expsm": E["expsm"],
                  "linear_extrap": (bt["K"] if f is None else None)}
            ea["linear_extrap"] = bt["K"] * 1e-9 if f is None else E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])
            sel = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], arms.index(a))) if f else bt["selected"]
            if f is None: ea[sel] = bt["selected_macs"] * 1e-9
            vR, tR, eR = bt["val"][sel], bt["test"][sel]["rho"], ea[sel]
            L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
            lw = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "lightweight"}
            if integer: lw = {int(s): (v["val_q"], v["test_q"]) for s, v in ACCQ[ds]["seeds"].items()}
            cn = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "cnn_lstm"}
            scr = {k: (E[k] - eR) >= (1 - vR) * ER for k in ("lightweight", "cnn_lstm")}
            en = lambda rho, e: ES + e + (1 - rho) * ER
            caps_e = []
            for s in sorted(lw):
                opts = [("ref", 0.0, en(tR, eR))]
                for k, D in (("lightweight", lw), ("cnn_lstm", cn)):
                    if not scr[k] and s in D: opts.append((k, (D[s][0] - vR) * ER - (E[k] - eR), en(D[s][1], E[k])))
                b = max(opts, key=lambda o: o[1]); picks[b[0]] += 1; caps_e.append(b[2])
            e_ref = en(tR, eR); e_lw = st.median(en(v[1], E["lightweight"]) for v in lw.values()); e_cnn = st.median(en(v[1], E["cnn_lstm"]) for v in cn.values())
            for k, v in (("ref", e_ref), ("lw", e_lw), ("cnn", e_cnn), ("caps", st.median(caps_e)), ("hind", min(e_ref, e_lw, e_cnn))): tot[k] += v
        r = {k: 100 * (tot[k] / tot["ref"] - 1) for k in tot}
        print("%-9s %-5s CAPS %+5.1f%%  hindsight %+5.1f%%  CAPS above hindsight %.2f%%  always-lw %+6.1f%%  always-CNN %+8.1f%%  picks %s" % (
            name, "gross" if gross else "inc", r["caps"], r["hind"], 100 * (tot["caps"] / tot["hind"] - 1), r["lw"], r["cnn"], picks))
