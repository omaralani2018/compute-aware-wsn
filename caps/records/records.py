"""Version 1.6. Writes to files the numbers of Sections 6.11 to 6.15 that the original scripts printed, or that were
taken from their outputs into the tables by hand: Table 12, Table 13, the totals of Section 6.11, the derived numbers
of Section 6.14 (cycle spread, energy of the low-cost predictors, integer minus floating-point forecasts, smallest
power step at a run start, crossover radio cost, streams that stay positive on gross energy, sample interval of the
power traces, the second integer node against the first), and the check of Section 6.15 that charges the 99,488-MAC
CNN-LSTM at 6,440 MACs. It reads only shipped outputs, changes none of them, and writes records/*.json.
Run from caps/ after setup.sh:  python3 records/records.py   -> records/*.json"""
import os, json, glob, statistics as st
C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(C, "records")
L = lambda *p: json.load(open(os.path.join(C, *p)))
RUN = ("stage17", "run"); MS = L(*RUN, "ms_numbers.json"); K = MS["constants"]
ER, EOP, ES = K["E_R"], K["E_OP"], K["E_SENSE"]
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly", "rds_cpu_util", "iio_networkin",
      "twitter_aapl", "exchange_cpc", "occupancy"]
def bt(ds): return MS["datasets"][ds]["by_tau"]["0.1"]
def learned(ds, k): return sorted([e for e in L(*RUN, f"learned_{ds}.json")["entries"] if e["predictor"] == k], key=lambda e: e["seed"])
# ---- Table 12 ----
VH = L("adaptA", "val_headroom.json")["streams"]; G1 = {r[0]: r for r in L("adaptA", "gate1_summary.json")["rows"]}
t12 = {}
for ds in DS:
    b = bt(ds); mR, vR = b["selected_macs"], b["val"][b["selected"]]
    budget = mR + (1 - vR) * ER / EOP
    orc = L("adaptA", f"oracle_{ds}_lightweight.json")["rows"]
    t12[ds] = dict(reference=b["selected"], budget_macs=budget, cnn_screened=bool(99488 >= budget),
                   select_lightweight=sum(bool(r["crit_picks_L"]) for r in orc),
                   headroom_val=VH[ds]["val_headroom"], headroom_test=G1[ds][1])
# ---- Section 6.11 totals (seed medians, sensing included), two candidate sets ----
def en(rho, m): return ES + m * EOP + (1 - rho) * ER
T = {k: 0.0 for k in ["ref", "lw", "cnn", "caps", "hind3", "hind3_switch", "lw_ta", "hind4_with_ta", "hind4_switch"]}
for ds in DS:
    b = bt(ds); ref, mR, vR, tR = b["selected"], b["selected_macs"], b["val"][b["selected"]], b["test"][b["selected"]]["rho"]
    lw, cn = learned(ds, "lightweight"), learned(ds, "cnn_lstm"); ta = sorted(L("tolA", f"ta_{ds}.json")["entries"], key=lambda e: e["seed"])
    orc = L("adaptA", f"oracle_{ds}_lightweight.json"); n = orc["n_readings"]
    e_ref = en(tR, mR); e_lw = st.median(en(e["test_rho"], 208) for e in lw); e_cn = st.median(en(e["test_rho"], 99488) for e in cn)
    e_ta = st.median(en(e["test_rho"], 208) for e in ta); e_sw = st.median(r["B96"]["E"] / n + ES for r in orc["rows"])
    screened = (99488 - mR) * EOP >= (1 - vR) * ER
    caps = []
    for i in range(20):
        opts = [(0.0, e_ref), ((lw[i]["val_rho"] - vR) * ER - (208 - mR) * EOP, en(lw[i]["test_rho"], 208))]
        if not screened: opts.append(((cn[i]["val_rho"] - vR) * ER - (99488 - mR) * EOP, en(cn[i]["test_rho"], 99488)))
        caps.append(max(opts, key=lambda o: o[0])[1])
    T["ref"] += e_ref; T["lw"] += e_lw; T["cnn"] += e_cn; T["caps"] += st.median(caps); T["lw_ta"] += e_ta
    T["hind3"] += min(e_ref, e_lw, e_cn); T["hind3_switch"] += min(e_ref, e_lw, e_cn, e_sw)
    T["hind4_with_ta"] += min(e_ref, e_lw, e_ta, e_cn); T["hind4_switch"] += min(e_ref, e_lw, e_ta, e_cn, e_sw)
rel = {k: v / T["ref"] - 1 for k, v in T.items()}
cnn_sw = max(r["B96"]["G_best"] for f in glob.glob(os.path.join(C, "adaptA", "oracle_*_cnn_lstm.json")) for r in json.load(open(f))["rows"])
totals = dict(note="relative to always running the validation-selected reference; hind3 = best per stream of the "
              "reference, the lightweight model and the CNN-LSTM; hind4 also counts the tolerance-aware model of Section 6.12",
              relative=rel, absolute_J=T, cnn_switching_max_saving=cnn_sw,
              gate1_streams_at_5pct=L("adaptA", "gate1_summary.json")["n_pass"])
# ---- Table 13 ----
TA, CLT = L("tolA", "ta_summary.json"), L("cltA", "clt_summary.json")
t13 = {ds: dict(rho_ref=TA["per_stream"][ds]["rho_R"], mse=TA["per_stream"][ds]["mse_med"], ta=TA["per_stream"][ds]["ta_med"],
                ta_sig_increase=TA["per_stream"][ds]["H1"], ta_saves=TA["per_stream"][ds]["H2_ta"],
                mse_saves=TA["per_stream"][ds]["H2_mse"], clt=CLT["per_stream"][ds]["ta_med"],
                clt_sig_increase=CLT["per_stream"][ds]["H1"], clt_saves=CLT["per_stream"][ds]["H2_ta"]) for ds in DS}
# ---- Section 6.14 derived numbers ----
PF = {"ATmega128L": L("avrsim", "avr_summary.json"), "M3 float": L("iotlab", "iotlab_m3_summary.json"),
      "M3 integer": L("iotlab", "fixed", "fixed_m3_21_accuracy_q_summary.json"), "nRF52840": L("iotlab", "iotlab_nrf_summary.json")}
LWK = {"ATmega128L": "lightweight", "M3 float": "lightweight", "M3 integer": "light_q", "nRF52840": "lightweight"}
CNK = {"ATmega128L": "cnn_lstm", "M3 float": "cnn_lstm", "M3 integer": "cnn_q", "nRF52840": "cnn_lstm"}
def E(pl, k):
    v = PF[pl]["per_forecast"][k]; return v["E_inc_J"] if "E_inc_J" in v else v["E_uJ"] * 1e-6
spread = {pl: {k: v["cycles_max"] / v["cycles_mean"] - 1 for k, v in J["per_forecast"].items() if v["cycles_mean"] > 0} for pl, J in PF.items()}
lowcost = {pl: {k: E(pl, k) for k in J["per_forecast"] if k not in (LWK[pl], CNK[pl])} for pl, J in PF.items()}
REFQ = L("iotlab", "fixed", "ref_q.json")
AQ = L("iotlab", "fixed", "accuracy_q.json"); cross = {}
for pl in PF:
    s = PF[pl]["streams"]["taxi"]; b = bt("taxi"); tR = b["test"][s["reference"]]["rho"]
    rho = [AQ["taxi"]["seeds"][str(i)]["test_q"] for i in range(20)] if pl == "M3 integer" else [e["test_rho"] for e in learned("taxi", "lightweight")]
    d = st.median(rho) - tR; cross[pl] = dict(reference=s["reference"], delta_rho=d, er_star_J=(E(pl, LWK[pl]) - s["E_ref_uJ"] * 1e-6) / d)
qerr = {k: max(abs(o / 2.0 ** REFQ["fi"] - f) for o, f in zip(REFQ["out"][k], REFQ["float_ref"][k])) for k in REFQ["float_ref"]}
derived = dict(cycle_spread_max_over_mean=spread, low_cost_energy_J=lowcost, crossover_taxi_lightweight=cross,
               anchor_er_J=ER, integer_minus_float_max_abs_degC_100_windows=qerr,
               smallest_power_step_at_run_start_W={pl: min(J["edge"]) for pl, J in PF.items() if "edge" in J})
# Section 6.14, gross energy (run power times time): streams where the lightweight model's median net saving stays
# positive, with the reference re-selected by gross energy as in capsplat/caps_platforms.py
GROSS = {}
for pl in ("M3 float", "M3 integer", "nRF52840"):
    pf = PF[pl]["per_forecast"]; m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
    Eg = {m.get(k, k): v["E_gross_J"] for k, v in pf.items()}; pos = {}
    for ds in DS:
        b = bt(ds); arms = ["persistence", "expsm", "linear_extrap"]
        ea = {"persistence": Eg["persistence"], "expsm": Eg["expsm"], "linear_extrap": Eg["lin2"] + (b["K"] - 2) / 10 * (Eg["lin12"] - Eg["lin2"])}
        sel = min(arms, key=lambda a: ((1 - b["val"][a]) * ER + ea[a], arms.index(a))); tR, eR = b["test"][sel]["rho"], ea[sel]
        rho = [AQ[ds]["seeds"][str(i)]["test_q"] for i in range(20)] if pl == "M3 integer" else [e["test_rho"] for e in learned(ds, "lightweight")]
        net = st.median((r - tR) * ER - (Eg["lightweight"] - eR) for r in rho)
        if net > 0: pos[ds] = net
    GROSS[pl] = pos
derived["gross_energy_positive_median_net_J"] = GROSS
# Section 6.14: sampling interval of the power traces (median spacing of the timestamps, columns 4 and 5) and the
# ratio of the second node's incremental energy to the first one's for the integer code
TR = {}
for tf in ("iotlab/data/m3_cons_cut2.txt", "iotlab/data/m3b_cons_cut.txt", "iotlab/data/nrf_cons_cut.txt",
            "iotlab/data/nrfb_cons_cut.txt", "iotlab/fixed/data/fix_m3_21.txt", "iotlab/fixed/data/fix_m3_22.txt"):
    t = [int(l.split()[3]) + int(l.split()[4]) * 1e-6 for l in open(os.path.join(C, tf)) if l.strip()]
    TR[tf] = st.median(b - a for a, b in zip(t, t[1:]))
derived["power_trace_median_sample_interval_s"] = TR
Q2 = L("iotlab", "fixed", "fixed_m3_22_accuracy_q_summary.json")["per_forecast"]
derived["integer_second_over_first_node_E_inc"] = {k: Q2[k]["E_inc_J"] / PF["M3 integer"]["per_forecast"][k]["E_inc_J"] for k in ("light_q", "cnn_q")}
# ---- Section 6.15: the 99,488-MAC CNN-LSTM charged at 6,440 MACs ----
chk = {}
for ds in DS:
    b = bt(ds); tR, mR = b["test"][b["selected"]]["rho"], b["selected_macs"]
    chk[ds] = st.median((e["test_rho"] - tR) * ER - (6440 - mR) * EOP for e in learned(ds, "cnn_lstm"))
for name, obj in [("table12", t12), ("totals_611", totals), ("table13", t13), ("platform_numbers", derived),
                  ("cnn_at_6440_macs", dict(note="median net saving (J) at er = 3.8e-4 J and 1 nJ per MAC", median_net_J=chk))]:
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=1, sort_keys=True)
print("Section 6.11 totals vs reference: " + ", ".join("%s %+.1f%%" % (k, 100 * v) for k, v in rel.items()))
print("max CNN-LSTM switching saving %.2f%%; Gate 1: %d streams at 5%%" % (100 * cnn_sw, totals["gate1_streams_at_5pct"]))
print("crossover on taxi: " + ", ".join("%s %.2e J" % (k, v["er_star_J"]) for k, v in cross.items()))
print("gross energy, positive median net: " + "; ".join("%s %s" % (k, sorted(v)) for k, v in GROSS.items()))
print("power trace sample interval (ms): " + ", ".join("%.2f" % (1e3 * v) for v in TR.values()))
print("CNN-LSTM at 6,440 MACs, positive median net (uJ): " + ", ".join("%s %.1f" % (k, 1e6 * v) for k, v in chk.items() if v > 0))
