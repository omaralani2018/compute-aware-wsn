import re, json, struct, statistics as st, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
raw = open(os.path.join(HERE, "raw_out.txt"), errors="replace").read()
rows = re.findall(r"(persistence|expsm|lin2|lin12|lightweight|cnn_lstm) (\d+) ([0-9a-f]{8})", raw)
base = int(re.search(r"base (\d+)", raw).group(1))
ref = json.load(open(os.path.join(HERE, "ref.json")))
V, I, F = 3.0, 8e-3, 7372800.0; EPC = V * I / F                      # J per cycle
MACS = {"persistence": 0, "expsm": 12, "lin2": 2, "lin12": 12, "lightweight": 208, "cnn_lstm": 99488}
res = {}
for k in MACS:
    c = [int(x) for n, x, _ in rows if n == k]; out = [struct.unpack(">f", bytes.fromhex(b))[0] for n, _, b in rows if n == k]
    assert len(c) == 100, (k, len(c))
    r = np.array(ref[k]); rel = np.max(np.abs(np.array(out) - r) / np.maximum(np.abs(r), 1e-12))
    res[k] = dict(cycles_mean=float(np.mean(c)), cycles_max=int(max(c)), max_rel_err=float(rel), ok=bool(rel < 1e-4),
                  E_uJ=float(np.mean(c) * EPC * 1e6), E_uJ_range=[float(np.mean(c) * 2.7 * I / F * 1e6), float(np.mean(c) * 3.3 * I / F * 1e6)],
                  nJ_per_MAC=float(np.mean(c) * EPC * 1e9 / MACS[k]) if MACS[k] else None, ms=float(np.mean(c) / F * 1e3))
    print("%-12s cycles %10.0f (max %9d)  %8.3f ms  E %9.3f uJ  [%s nJ/MAC]  max rel err %.1e %s" % (
        k, res[k]["cycles_mean"], res[k]["cycles_max"], res[k]["ms"], res[k]["E_uJ"],
        "%.0f" % res[k]["nJ_per_MAC"] if res[k]["nJ_per_MAC"] else "-", rel, "OK" if res[k]["ok"] else "MISMATCH"))
print("timing overhead subtracted: %d cycles; energy per cycle %.3f nJ" % (base, EPC * 1e9))
# CAPS at the anchor with these per-forecast energies
ER = 3.8e-4; MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
def e_arm(name, K=None):
    if name == "persistence": return res["persistence"]["E_uJ"] * 1e-6
    if name == "expsm": return res["expsm"]["E_uJ"] * 1e-6
    if name == "linear_extrap": return (res["lin2"]["E_uJ"] + (K - 2) / 10 * (res["lin12"]["E_uJ"] - res["lin2"]["E_uJ"])) * 1e-6
out = {"per_forecast": res, "streams": {}}; sav = {"lightweight": 0, "cnn_lstm": 0}
for ds, d in MS.items():
    bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
    E = {a: e_arm(a, bt["K"]) for a in arms}
    sel = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + E[a], arms.index(a)))
    tR = bt["test"][sel]["rho"]; vR = bt["val"][sel]
    L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
    row = dict(reference=sel, reference_paper=bt["selected"], E_ref_uJ=E[sel] * 1e6)
    for k in ("lightweight", "cnn_lstm"):
        Ek = res[k]["E_uJ"] * 1e-6
        net = [(e["test_rho"] - tR) * ER - (Ek - E[sel]) for e in L if e["predictor"] == k]
        row[k] = dict(median_net_uJ=st.median(net) * 1e6, seeds_positive=sum(x > 0 for x in net), screened=(Ek - E[sel]) >= (1 - vR) * ER)
        sav[k] += st.median(net) > 0
    out["streams"][ds] = row
    print("%-17s ref %-13s (paper %-13s) | lightweight net %+8.1f uJ (%2d/20 >0) screened %-5s | CNN-LSTM net %+10.1f uJ screened %s" % (
        ds, sel, bt["selected"], row["lightweight"]["median_net_uJ"], row["lightweight"]["seeds_positive"], row["lightweight"]["screened"],
        row["cnn_lstm"]["median_net_uJ"], row["cnn_lstm"]["screened"]))
print("streams with positive median net saving: lightweight %d/11, CNN-LSTM %d/11" % (sav["lightweight"], sav["cnn_lstm"]))
json.dump(out, open(os.path.join(HERE, "avr_summary.json"), "w"), indent=1)
