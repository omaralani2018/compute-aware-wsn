"""Record written for version 1.7: compute-blind picks per dataset on the twelve further datasets (the paper's
"at most 4 of the 20 seeds of any dataset"). Same selection as blind_conf.py; writes picks_conf.json."""
import os, sys, json, glob
SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAC = {"lightweight": 208, "cnn_lstm": 99488}
def seeds_from_learned(f):
    L = json.load(open(f))["entries"]; return {k: {e["seed"]: e["val_rho"] for e in L if e["predictor"] == k} for k in MAC}
def seeds_from_runs(d):
    out = {k: {} for k in MAC}
    for f in glob.glob(os.path.join(d, "runs", "stb_obs_*_s*.json")):
        r = json.load(open(f))
        if r.get("status") == "ok": out[r["predictor"]][r["seed"]] = r["val_rho"]
    return out
groups = [("confA/conf_summary.json", "streams", lambda n: seeds_from_learned(os.path.join(SC, "confA", f"learned_{n}.json"))),
          ("confB/confB_summary.json", "deployments", lambda n: seeds_from_runs(os.path.join(SC, "confB", f"run_{n}"))),
          ("confC/conf_summary.json", "streams", lambda n: seeds_from_learned(os.path.join(SC, "confC", f"learned_{n}.json")))]
out = {}
for f, key, getter in groups:
    for n, s in json.load(open(os.path.join(SC, f)))[key].items():
        D = getter(n); vR = s["rho_ref_val"]; c = {}
        for i in range(20):
            bo = [("ref", vR, 0)] + [(k, D[k][i], MAC[k]) for k in MAC]
            b = max(bo, key=lambda o: (o[1], -o[2]))[0]; c[b] = c.get(b, 0) + 1
        out[n] = c
tot = {}
for c in out.values():
    for k, v in c.items(): tot[k] = tot.get(k, 0) + v
res = {"per_dataset": out, "total": tot, "max_cnn_lstm_seeds_any_dataset": max(c.get("cnn_lstm", 0) for c in out.values())}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "picks_conf.json"), "w"), indent=1)
print(res["total"], "max CNN-LSTM seeds on one dataset:", res["max_cnn_lstm_seeds_any_dataset"])
