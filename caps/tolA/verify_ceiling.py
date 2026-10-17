# Independent recomputation (written separately from caps_vs_fixed.py).
# Per-round energy as in the paper: Esense + Einf + (1-rho)*er.
import json, statistics as st
RUN = "stage17/run"; ms = json.load(open(RUN + "/ms_numbers.json"))
C = ms["constants"]; er, eop, es = C["E_R"], C["E_OP"], C["E_SENSE"]
def en(rho, macs): return es + macs * eop + (1 - rho) * er
names = ["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]
T = {k: 0.0 for k in ["ref", "lw", "lw_ta", "cnn", "caps", "caps_ta", "hind_fixed", "hind_all"]}
for s in names:
    b = ms["datasets"][s]["by_tau"]["0.1"]; ref = b["selected"]; mref = b["selected_macs"]
    vref, tref = b["val"][ref], b["test"][ref]["rho"]
    ent = json.load(open(f"{RUN}/learned_{s}.json"))["entries"]
    lw = sorted([e for e in ent if e["predictor"] == "lightweight"], key=lambda e: e["seed"])
    cn = sorted([e for e in ent if e["predictor"] == "cnn_lstm"], key=lambda e: e["seed"])
    ta = sorted(json.load(open(f"tolA/ta_{s}.json"))["entries"], key=lambda e: e["seed"])
    orc = json.load(open(f"adaptA/oracle_{s}_lightweight.json")); n = orc["n_readings"]
    e_ref = en(tref, mref)
    e_lw = st.median(en(e["test_rho"], 208) for e in lw)
    e_ta = st.median(en(e["test_rho"], 208) for e in ta)
    e_cn = st.median(en(e["test_rho"], 99488) for e in cn)
    screened = (99488 - mref) * eop >= (1 - vref) * er
    def caps(lwset):
        out = []
        for i in range(20):
            opts = [(0.0, e_ref), ((lwset[i]["val_rho"] - vref) * er - (208 - mref) * eop, en(lwset[i]["test_rho"], 208))]
            if not screened: opts.append(((cn[i]["val_rho"] - vref) * er - (99488 - mref) * eop, en(cn[i]["test_rho"], 99488)))
            out.append(max(opts, key=lambda o: o[0])[1])
        return st.median(out)
    e_sw = st.median(r["B96"]["E"] / n + es for r in orc["rows"])      # hindsight block switching R<->lw
    T["ref"] += e_ref; T["lw"] += e_lw; T["lw_ta"] += e_ta; T["cnn"] += e_cn
    T["caps"] += caps(lw); T["caps_ta"] += caps(ta)
    T["hind_fixed"] += min(e_ref, e_lw, e_ta, e_cn); T["hind_all"] += min(e_ref, e_lw, e_ta, e_cn, e_sw)
for k, v in T.items(): print("%-10s %.4e  (%+.1f%% vs ref)" % (k, v, 100 * (v / T["ref"] - 1)))
print("cnn costs %.1f%% more than caps_ta" % (100 * (T["cnn"] / T["caps_ta"] - 1)))
