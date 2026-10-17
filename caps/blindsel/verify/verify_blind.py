"""Independent re-implementation of PREREG16 configs A (subset) and B: CAPS vs compute-blind vs always-reference."""
import os, json, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # caps/
J = lambda f: json.load(open(os.path.join(SC, f)))
ES = 1e-7
MS = J("stage17/run/ms_numbers.json")["datasets"]
ACCQ = J("iotlab/fixed/accuracy_q.json")
ARMS = ["persistence", "expsm", "linear_extrap"]
RENAME = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
PLATS = {"mica2": "avrsim/avr_summary.json", "m3_float": "iotlab/iotlab_m3_summary.json",
         "m3_int": "iotlab/fixed/fixed_m3_21_accuracy_q_summary.json", "nrf": "iotlab/iotlab_nrf_summary.json"}
MACS_LW, MACS_CNN = 208, 99488

def plat_E(f):
    pf = J(f)["per_forecast"]
    return {RENAME.get(k, k): (v["E_inc_J"] if "E_inc_J" in v else v["E_uJ"] * 1e-6) for k, v in pf.items()}

def run(plat, er, agg):
    anchor = plat is None
    E = None if anchor else plat_E(PLATS[plat])
    integer = plat == "m3_int"
    tot = {"caps": 0.0, "blind": 0.0, "ref": 0.0}; per = {}
    for ds, d in MS.items():
        bt = d["by_tau"]["0.1"]
        def arm_E(a):
            if anchor: return {"persistence": 0.0, "expsm": 12e-9, "linear_extrap": bt["K"] * 1e-9}[a]
            if a == "linear_extrap": return E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])
            return E[a]
        eL = 208e-9 if anchor else E["lightweight"]; eC = 99488e-9 if anchor else E["cnn_lstm"]
        # blind reference: highest validation suppression, ties to fewer MACs (MACs: pers 0, expsm 12, lin K)
        mac = {"persistence": 0, "expsm": 12, "linear_extrap": bt["K"]}
        rb = max(ARMS, key=lambda a: (bt["val"][a], -mac[a]))
        assert rb == bt["selected"], (ds, rb, bt["selected"])
        # CAPS reference: anchor -> val suppression; platform -> min validation energy (ties: list order)
        rc = rb if anchor else min(ARMS, key=lambda a: ((1 - bt["val"][a]) * er + arm_E(a), ARMS.index(a)))
        eRb = bt["selected_macs"] * 1e-9 if anchor else arm_E(rb)
        eRc = bt["selected_macs"] * 1e-9 if anchor else arm_E(rc)
        en = lambda rho, e: ES + e + (1 - rho) * er
        L = J(f"stage17/run/learned_{ds}.json")["entries"]
        lw = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "lightweight"}
        if integer: lw = {int(s): (v["val_q"], v["test_q"]) for s, v in ACCQ[ds]["seeds"].items()}
        cn = {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == "cnn_lstm"}
        seeds = sorted(set(lw) & set(cn)); assert len(seeds) == 20
        # CAPS
        vR, tR = bt["val"][rc], bt["test"][rc]["rho"]
        excl = {"lw": (eL - eRc) >= (1 - vR) * er, "cnn": (eC - eRc) >= (1 - vR) * er}
        caps, blind, ncnn = [], [], 0
        for s in seeds:
            c = [((0.0), en(tR, eRc))]
            if not excl["lw"]: c.append(((lw[s][0] - vR) * er - (eL - eRc), en(lw[s][1], eL)))
            if not excl["cnn"]: c.append(((cn[s][0] - vR) * er - (eC - eRc), en(cn[s][1], eC)))
            caps.append(max(c, key=lambda x: x[0])[1])  # first max wins (ref on ties)
            b = [(bt["val"][rb], -bt["selected_macs"] if False else -mac[rb], en(bt["test"][rb]["rho"], eRb), "ref"),
                 (lw[s][0], -MACS_LW, en(lw[s][1], eL), "lw"), (cn[s][0], -MACS_CNN, en(cn[s][1], eC), "cnn")]
            pick = max(b, key=lambda x: (x[0], x[1])); blind.append(pick[2]); ncnn += pick[3] == "cnn"
        ref_e = en(tR, eRc)
        a = {"caps": agg(caps), "blind": agg(blind), "ref": ref_e}
        for k in tot: tot[k] += a[k]
        per[ds] = (a["blind"] / a["caps"], ncnn)
    return tot, per

if __name__ == "__main__":
    out = {}
    for agg_name, agg in (("median", st.median), ("mean", st.mean)):
        cfgs = [("1nJ", None, er) for er in (1e-4, 3.8e-4, 3e-3)] + [(p, p, 3.8e-4) for p in PLATS]
        for name, plat, er in cfgs:
            tot, per = run(plat, er, agg)
            bc = 100 * (tot["blind"] / tot["caps"] - 1); cr = 100 * (tot["caps"] / tot["ref"] - 1); br = 100 * (tot["blind"] / tot["ref"] - 1)
            drv = {ds: (round(r, 3), n) for ds, (r, n) in per.items() if r > 1.05}
            ncnn = {ds: n for ds, (r, n) in per.items() if n}
            out[f"{agg_name}|{name}|{er}"] = dict(blind_vs_caps=bc, caps_vs_ref=cr, blind_vs_ref=br, tot=tot, drivers=drv, cnn_picks=ncnn)
            print(f"{agg_name:6s} {name:9s} er={er:g}  blind vs CAPS {bc:+7.2f}%  CAPS vs ref {cr:+6.2f}%  blind vs ref {br:+7.2f}%")
            if agg_name == "median": print("     drivers(ratio,nCNN):", drv, " CNN picks:", ncnn)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_out.json"), "w"), indent=1)
