"""Intel preparation variants for the preprocessing-sensitivity analysis.

Rules R1-R4 are those of prepare_intel.py (mote ids 1-54, 0 < T < 50 degC, 31-second bins from
2004-02-28 00:00:00, motes with at most 50% missing bins over their own active span), so the same
30 motes are kept. The published matrix (V0) then interpolates up to 8 bins of every interior gap
before the split and keeps complete rows. The two variants below instead:

  * use common chronological split boundaries fixed BEFORE any filling: the calendar boundaries of
    the published split (validation from 2004-03-04 00:11:07, test from 2004-03-06 03:29:25) and
    its first and last row (2004-02-28 01:09:45 to 2004-03-08 21:29:09);
  * evaluate each mote on its own series, cut into segments at every gap that is not filled and at
    each split boundary, so no value and no predictor history crosses a boundary or a long gap;
  * a gap is short if the elapsed time between the readings that bracket it is at most SHORT_GAP_S
    seconds (mean timestamps of the readings in the two bracketing bins); a longer gap ends a segment;
  * V_obs: observed readings only, nothing filled; a short gap is skipped, so the predictor's next
    forecast is for the next reading (the history is the sequence of the mote's own readings);
  * V_short: the bins of each short gap are filled by linear interpolation in elapsed time.
  A stricter observed-only rule, ending a segment at every missing 31-second bin, is reported in the
  coverage only: about a quarter of each mote's bins are empty, so its segments have a median length of
  two readings and almost every reading would be a warm-up report.

Each segment is later evaluated from its own start: its first LOOK_BACK readings are sent and
charged as initialisation reports, as the warm-up of the main study is.
"""
import pandas as pd, numpy as np, hashlib, json, pickle, sys
SRC = "raw/data.txt"; BIN = "31s"; ORIGIN = "2004-02-28 00:00:00"
FIRST, VAL0, TEST0, LAST = (pd.Timestamp(x) for x in
    ("2004-02-28 01:09:45", "2004-03-04 00:11:07", "2004-03-06 03:29:25", "2004-03-08 21:29:09"))
SHORT_GAP_S = 300.0
LOOK_BACK = 12

def load_bins():
    cols = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
    raw = pd.read_csv(SRC, sep=" ", header=None, names=cols).dropna(subset=["moteid"]); raw["moteid"] = raw.moteid.astype(int)
    d = raw[(raw.moteid >= 1) & (raw.moteid <= 54)]
    d = d[(d.temperature > 0) & (d.temperature < 50)].dropna(subset=["temperature"])
    d = d.assign(ts=pd.to_datetime(d.date + " " + d.time, errors="coerce")).dropna(subset=["ts"])
    origin = pd.Timestamp(ORIGIN); d["bin"] = origin + ((d.ts - origin) // pd.Timedelta(BIN)) * pd.Timedelta(BIN)
    p = d.pivot_table(index="bin", columns="moteid", values="temperature", aggfunc="mean")
    d["tsn"] = (d.ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    pt = d.pivot_table(index="bin", columns="moteid", values="tsn", aggfunc="mean")     # mean reading time per bin
    grid = pd.date_range(p.index.min(), p.index.max(), freq=BIN); p = p.reindex(grid); pt = pt.reindex(grid)
    own = pd.Series({m: p[m].loc[p[m].first_valid_index():p[m].last_valid_index()].isna().mean() for m in p.columns})
    keep = p.columns[own <= 0.5]
    return p[keep], pt[keep]

def spans_of(index):
    lab = np.full(len(index), "", dtype=object)
    lab[(index >= FIRST) & (index < VAL0)] = "train"
    lab[(index >= VAL0) & (index < TEST0)] = "val"
    lab[(index >= TEST0) & (index <= LAST)] = "test"
    return lab

def segments(p, pt, variant):
    """-> list of dicts(mote, span, bins (bin start, seconds since 1970), values (float), filled (bool)).
    Both variants share the same segments: a segment ends at a split boundary and wherever the elapsed
    time between consecutive readings of the mote exceeds SHORT_GAP_S. V_obs keeps only the observed
    readings of a segment (a short gap is skipped, so the next forecast is for the next reading);
    V_short fills the bins of each short gap by linear interpolation in elapsed time."""
    out = []
    lab = spans_of(p.index)
    secs = (p.index - pd.Timestamp("1970-01-01")).total_seconds().to_numpy()
    for m in p.columns:
        for sp in ("train", "val", "test"):
            sel = lab == sp
            v = p[m].to_numpy()[sel]; t = pt[m].to_numpy()[sel]; b = secs[sel]
            i = np.flatnonzero(~np.isnan(v))
            if len(i) == 0: continue
            brk = np.flatnonzero((t[i[1:]] - t[i[:-1]]) > SHORT_GAP_S)
            for run in np.split(i, brk + 1):
                if variant == "obs":
                    k = run; vals = v[k].astype(float); fl = np.zeros(len(k), bool)
                else:
                    k = np.arange(run[0], run[-1] + 1); vals = v[k].astype(float); fl = np.isnan(vals)
                    if fl.any():
                        tb = b[k] + 15.5                               # bin centre, seconds
                        tt = np.where(fl, tb, t[k])
                        vals[fl] = np.interp(tt[fl], t[run], v[run])
                out.append(dict(mote=int(m), span=sp, bins=b[k], values=vals, filled=fl))
    return out

def coverage(segs):
    c = {}
    for sp in ("train", "val", "test"):
        S = [s for s in segs if s["span"] == sp]
        n = sum(len(s["values"]) for s in S); usable = [s for s in S if len(s["values"]) > LOOK_BACK]
        c[sp] = dict(segments=len(S), cells=int(n), cells_filled=int(sum(s["filled"].sum() for s in S)),
                     segments_longer_than_lookback=len(usable),
                     cells_in_usable_segments=int(sum(len(s["values"]) for s in usable)),
                     warmup_cells=int(sum(min(len(s["values"]), LOOK_BACK) for s in S)),
                     closed_loop_steps=int(sum(len(s["values"]) - LOOK_BACK for s in usable)),
                     median_segment_len=float(np.median([len(s["values"]) for s in S])) if S else 0)
    return c

if __name__ == "__main__":
    p, pt = load_bins()
    assert list(p.columns) == json.load(open("intel_time_meta.json"))["motes"]
    res = {"source_sha256": hashlib.sha256(open(SRC, "rb").read()).hexdigest(), "motes": [int(m) for m in p.columns],
           "boundaries": [str(FIRST), str(VAL0), str(TEST0), str(LAST)], "short_gap_s": SHORT_GAP_S, "variants": {}}
    for var in ("obs", "short"):
        S = segments(p, pt, var)
        pickle.dump(S, open(f"intel_segments_{var}.pkl", "wb"))
        res["variants"][var] = coverage(S)
    # strict observed-only rule (a segment ends at every missing bin): coverage only
    lab = spans_of(p.index); st = {}
    for sp in ("train", "val", "test"):
        L = []
        for m in p.columns:
            v = p[m].to_numpy()[lab == sp]; i = np.flatnonzero(~np.isnan(v))
            if len(i): L += [len(r) for r in np.split(i, np.flatnonzero(np.diff(i) > 1) + 1)]
        L = np.array(L)
        st[sp] = dict(segments=int(len(L)), cells=int(L.sum()), median_segment_len=float(np.median(L)),
                      closed_loop_steps=int(np.clip(L - LOOK_BACK, 0, None).sum()))
    res["strict_contiguous_coverage_only"] = st
    # V0 coverage for comparison: observed (not interpolated) cells in each span of the published matrix
    lab = spans_of(p.index)
    rows = pd.to_datetime(pd.read_csv("intel_time_rows.csv")["bin_start"])
    M = np.load("intel_time.npy"); P = p.reindex(rows)
    obs0 = ~np.isnan(P.to_numpy())
    tr, va = int(.5 * len(M)), int(.7 * len(M))
    res["V0"] = {sp: dict(cells=int(M[a:b].size), cells_observed=int(obs0[a:b].sum())) for sp, (a, b) in
                 {"train": (0, tr), "val": (tr, va), "test": (va, len(M))}.items()}
    assert np.allclose(M[obs0], P.to_numpy()[obs0])          # observed cells of V0 equal the bin means
    json.dump(res, open("intel_variants_coverage.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
