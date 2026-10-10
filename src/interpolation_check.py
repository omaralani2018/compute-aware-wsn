"""What the interpolation step R5 of prepare_intel.py does to the paper's Intel matrix (Section 5.2).
Rebuilds the matrix with the same code as prepare_intel.py (R1-R6), asserts it is identical to
intel_time.npy, and reports:
  - how many cells of the final matrix were filled by interpolation, and how many of them come from
    interior gaps longer than eight bins (pandas interpolate(limit=8) fills the first eight missing
    bins of every interior gap, whatever its length; the rest of a longer gap stays missing and its
    rows are dropped by R6);
  - for those longer gaps: their number, their lengths, and how far the filled values lie from the
    last reading before the gap;
  - for each split boundary (50/20/30 by row), the rows holding an interpolated value whose gap has
    readings on both sides of the boundary, i.e. a value that depends on a reading across it.
Inputs: raw/data.txt, intel_time.npy, intel_time_rows.csv. Output: interpolation_check.json."""
import json, numpy as np, pandas as pd
SRC = "raw/data.txt"; BIN = "31s"; ORIGIN = "2004-02-28 00:00:00"; LIMIT = 8
cols = ["date","time","epoch","moteid","temperature","humidity","light","voltage"]
raw = pd.read_csv(SRC, sep=" ", header=None, names=cols).dropna(subset=["moteid"]); raw["moteid"] = raw.moteid.astype(int)
d = raw[(raw.moteid >= 1) & (raw.moteid <= 54)]                                        # R1
d = d[(d.temperature > 0) & (d.temperature < 50)].dropna(subset=["temperature"])      # R2
d = d.assign(ts=pd.to_datetime(d.date+" "+d.time, errors="coerce")).dropna(subset=["ts"])
origin = pd.Timestamp(ORIGIN); d["bin"] = origin+((d.ts-origin)//pd.Timedelta(BIN))*pd.Timedelta(BIN)   # R3
p = d.pivot_table(index="bin", columns="moteid", values="temperature", aggfunc="mean")
p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq=BIN))
own = pd.Series({m: p[m].loc[p[m].first_valid_index():p[m].last_valid_index()].isna().mean() for m in p.columns})
p = p[p.columns[own <= 0.5]]                                                            # R4
pi = p.interpolate(limit=LIMIT, limit_area="inside")                                    # R5
q = pi.dropna()                                                                         # R6
assert np.array_equal(q.to_numpy(), np.load("intel_time.npy")), "rebuilt matrix differs from intel_time.npy"
assert (q.index == pd.to_datetime(pd.read_csv("intel_time_rows.csv")["bin_start"])).all(), "row times differ from intel_time_rows.csv"
T = len(q); tr, va = int(.5*T), int(.7*T); bounds = {"train_validation": q.index[tr], "validation_test": q.index[va]}
rows = set(q.index); idx = p.index
filled_total = 0; long_cells = []; long_rows = set(); long_gaps = []; n_long_interior = 0
cross = {k: set() for k in bounds}; cross_long = 0; reach = {k: 0.0 for k in bounds}
for c in p.columns:
    na = p[c].isna().to_numpy(); v = p[c].to_numpy(); i = 0
    while i < len(na):
        if not na[i]: i += 1; continue
        j = i
        while j < len(na) and na[j]: j += 1
        if i > 0 and j < len(na):                                    # interior gap idx[i:j]
            L = j - i; a, b = idx[i-1], idx[j]; used = False; n_long_interior += int(L > LIMIT)
            for k in range(i, min(i+LIMIT, j)):
                t = idx[k]
                if t not in rows: continue
                filled_total += 1
                if L > LIMIT:
                    long_cells.append(abs(pi[c].iloc[k]-v[i-1])); long_rows.add(t); used = True
                for name, bd in bounds.items():
                    if a < bd <= b:                                  # the value depends on a reading across bd
                        cross[name].add(t); cross_long += int(L > LIMIT)
                        far = b if t < bd else a                         # the reading on the other side
                        reach[name] = max(reach[name], abs((far - t).total_seconds()))
            if used: long_gaps.append(L)
        i = j
dev = np.array(long_cells); G = np.array(long_gaps)
off = {n: sorted(q.index.get_loc(t)-q.index.get_loc(bounds[n]) for t in s) for n, s in cross.items()}
out = dict(matrix_identical_to_intel_time=True, cells=int(q.size), rows=int(T),
           filled_cells=int(filled_total),
           interior_gaps_longer_than_8=int(n_long_interior),
           from_gaps_longer_than_8=dict(cells=int(len(dev)), rows=int(len(long_rows)), gaps_contributing_kept_cells=int(len(G)),
               gap_length_bins=dict(min=int(G.min()), median=float(np.median(G)), max=int(G.max())),
               abs_diff_from_last_reading_degC=dict(median=float(np.median(dev)), p95=float(np.percentile(dev, 95)), max=float(dev.max()))),
           split_boundaries={n: dict(first_row_of_later_span=str(bounds[n]), rows_with_cross_boundary_value=len(cross[n]),
                                     rows_before_boundary=sum(o < 0 for o in off[n]), row_offsets=off[n],
                                     max_seconds_row_to_boundary=float(max(abs((t-bounds[n]).total_seconds()) for t in cross[n])),
                                     max_seconds_value_to_reading_across=reach[n]) for n in bounds},
           cross_boundary_cells_from_gaps_longer_than_8=int(cross_long))
json.dump(out, open("interpolation_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
