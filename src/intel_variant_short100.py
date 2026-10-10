"""Third Intel variant for the sensitivity analysis, V_short100: the segments of V_obs and V_short
(cut at the calendar split boundaries and at gaps longer than 300 s), with only genuinely short gaps
filled: a gap is interpolated (linearly in elapsed time) if the readings that bracket it are at most
100 s apart (at most two missing 31-s bins); a longer gap inside a segment is skipped, as in V_obs.
Writes intel_segments_short100.pkl and adds its coverage to intel_variants_coverage.json."""
import json, pickle, numpy as np, pandas as pd
import intel_variants as iv
FILL_S = 100.0
p, pt = iv.load_bins(); lab = iv.spans_of(p.index)
secs = (p.index - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
out = []
for m in p.columns:
    for sp in ("train", "val", "test"):
        sel = lab == sp; v = p[m].to_numpy()[sel]; t = pt[m].to_numpy()[sel]; b = secs[sel]
        i = np.flatnonzero(~np.isnan(v))
        if len(i) == 0: continue
        brk = np.flatnonzero((t[i[1:]] - t[i[:-1]]) > iv.SHORT_GAP_S)
        for run in np.split(i, brk + 1):
            k = [run[0]]
            for a, c in zip(run[:-1], run[1:]):
                if c - a > 1 and (t[c] - t[a]) <= FILL_S: k += list(range(a + 1, c))
                k.append(c)
            k = np.array(k); vals = v[k].astype(float); fl = np.isnan(vals)
            if fl.any():
                vals[fl] = np.interp(b[k][fl] + 15.5, t[run], v[run])
            out.append(dict(mote=int(m), span=sp, bins=b[k], values=vals, filled=fl))
pickle.dump(out, open("intel_segments_short100.pkl", "wb"))
C = json.load(open("intel_variants_coverage.json")); C["variants"]["short100"] = iv.coverage(out); C["short100_fill_s"] = FILL_S
json.dump(C, open("intel_variants_coverage.json", "w"), indent=1)
print(json.dumps(C["variants"]["short100"]["test"]))
