"""PREREG16 configuration C: compute-blind vs CAPS on the twelve confirmatory datasets (1 nJ/MAC, er = 3.8e-4 J).
Uses the confirmatory summaries (reference choice, validation suppression and energies) and the per-seed learned results."""
import os, sys, json, glob, statistics as st
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SC, "stage17/run")); import frozen_benchmark as fb
ER, EOP, ES = 3.8e-4, 1e-9, fb.E_SENSE
MAC = {"lightweight": 208, "cnn_lstm": 99488}
en = lambda rho, m: ES + m * EOP + (1 - rho) * ER
def seeds_from_learned(f):
    L = json.load(open(f))["entries"]; return {k: {e["seed"]: (e["val_rho"], e["test_rho"]) for e in L if e["predictor"] == k} for k in MAC}
def seeds_from_runs(d):
    out = {k: {} for k in MAC}
    for f in glob.glob(os.path.join(d, "runs", "stb_obs_*_s*.json")):
        r = json.load(open(f))
        if r.get("status") == "ok": out[r["predictor"]][r["seed"]] = (r["val_rho"], r["test_rho"])
    return out
groups = [("NAB", "confA/conf_summary.json", "streams", lambda n: seeds_from_learned(os.path.join(SC, "confA", f"learned_{n}.json"))),
          ("SensorScope", "confB/confB_summary.json", "deployments", lambda n: seeds_from_runs(os.path.join(SC, "confB", f"run_{n}"))),
          ("Indoor", "confC/conf_summary.json", "streams", lambda n: seeds_from_learned(os.path.join(SC, "confC", f"learned_{n}.json")))]
OUT = {}; T = {"caps": 0, "blind": 0, "ref": 0, "hind": 0}; P = {}
for g, f, key, getter in groups:
    S_ = json.load(open(os.path.join(SC, f)))[key]; tg = {"caps": 0, "blind": 0, "ref": 0, "hind": 0}
    for n, s in S_.items():
        D = getter(n); vR = s["rho_ref_val"]; eR = s["energy"]["ref"]
        assert all(len(D[k]) == 20 for k in MAC), (n, {k: len(D[k]) for k in MAC})
        be = []
        for i in range(20):
            bo = [("ref", vR, 0, eR)] + [(k, D[k][i][0], MAC[k], en(D[k][i][1], MAC[k])) for k in MAC]
            b = max(bo, key=lambda o: (o[1], -o[2])); P[b[0]] = P.get(b[0], 0) + 1; be.append(b[3])
        row = {"caps": s["energy"]["caps"], "blind": st.median(be), "ref": eR, "hind": s["energy"]["hind"]}
        OUT[n] = row
        for k in tg: tg[k] += row[k]; T[k] += row[k]
    print("%-12s blind vs CAPS %+6.1f%% | CAPS vs ref %+5.1f%% | blind vs ref %+6.1f%%" % (g, 100 * (tg["blind"] / tg["caps"] - 1), 100 * (tg["caps"] / tg["ref"] - 1), 100 * (tg["blind"] / tg["ref"] - 1)))
print("ALL twelve   blind vs CAPS %+6.1f%% | CAPS vs ref %+5.1f%% | blind picks %s" % (100 * (T["blind"] / T["caps"] - 1), 100 * (T["caps"] / T["ref"] - 1), P))
json.dump({"per": OUT, "tot": T, "blind_picks": P}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind_conf.json"), "w"), indent=1)
