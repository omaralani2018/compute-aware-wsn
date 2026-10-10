"""Descriptive uncertainty intervals for the per-unit effect sizes of Sections 6.9 and 6.10.
For each learned model: the per-unit mean suppression difference from the validation-selected reference
(Intel: per mote, from intel_mote_blocks.json; Grand St Bernard: per station, from stb_summary.json,
primary preparation), its mean over units with a 95% percentile bootstrap interval over units (10,000
resamples, numpy default_rng(0)), range, median and interquartile range; and per third of the test span the
mean difference with the number of seeds above the reference. Units share one environment (one room, one
pass), so they are not independent and the intervals are descriptive. Writes effect_intervals.json."""
import json, sys, numpy as np
STB = sys.argv[1] if len(sys.argv) > 1 else "stb_summary.json"
rng = np.random.default_rng(0)
def summ(x, B=10000):
    x = np.asarray(x, float); bm = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return dict(n=len(x), mean=float(x.mean()), ci95=[float(v) for v in np.percentile(bm, [2.5, 97.5])],
                min=float(x.min()), max=float(x.max()), median=float(np.median(x)), iqr=[float(np.percentile(x, 25)), float(np.percentile(x, 75))], n_positive=int((x > 0).sum()))
out = {"intel_per_mote": {}, "intel_per_third": {}, "gsb_per_station": {}, "gsb_per_third": {}}
B = json.load(open("intel_mote_blocks.json"))
for m, x in B["models"].items():
    out["intel_per_mote"][m] = summ([p["delta_mean"] for p in x["per_mote"]])
    out["intel_per_third"][m] = [dict(third=b["block"], delta_mean=b["delta_mean"], seeds_above_reference=b["seeds_above_reference"]) for b in x["per_block"]]
S = json.load(open(STB))["primary_obs"]["models"]
for m, x in S.items():
    out["gsb_per_station"][m] = summ([s["delta"] for s in x["per_station_descriptive"]["stations"]])
    out["gsb_per_third"][m] = [dict(third=b["block"], delta=b["delta"]) for b in x["per_block_descriptive"]]
json.dump(out, open("effect_intervals.json", "w"), indent=1)
for k, v in out.items():
    for m, r in v.items():
        print(k, m, {a: (round(b, 4) if isinstance(b, float) else ([round(c, 4) for c in b] if isinstance(b, list) and b and isinstance(b[0], float) else b)) for a, b in r.items()} if isinstance(r, dict) else r)
