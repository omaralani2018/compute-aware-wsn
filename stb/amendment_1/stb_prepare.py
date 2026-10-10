"""Preparation of the Grand St Bernard deployment (STB_PROTOCOL.md, section 3). Fixed before any
measurement or timestamp of the deployment was read.

Input : stbernard-meteo.zip (SensorScope archive, Zenodo 10.5281/zenodo.2654726): one whitespace-
        separated file per station, 17 columns (stbernard-meteo-def.txt); column 1 = station id,
        column 8 = time since the epoch [s], column 9 = ambient temperature [degC].
Output: stb_segments_obs.pkl, stb_segments_fill.pkl and stb_meta.json.
Usage : python3 stb_prepare.py stbernard-meteo.zip

Steps, in the order of the protocol (all automatic):
 1. Parsing, in file order: a line is used only if it has exactly 17 fields, field 1 equals the station
    number of its file, and fields 8 and 9 are finite numbers; other lines are dropped and counted (with
    a histogram of field counts). Timestamps outside 2005-01-01 to 2011-01-01 UTC (missing, wrong unit or
    implausible dates) are dropped and counted. A reading more than 1 hour earlier than the latest
    timestamp already accepted from the same file (a clock reset) is dropped and counted, unless this
    would drop more than 5% of the file's lines, in which case the file is taken as not written in time
    order, the rule is not applied to it, and this is reported. Temperatures
    outside the open interval (-50, 50) degC are dropped and counted; no other cleaning is done, and
    readings more than 5 degC from both neighbours are only counted. The readings are then sorted by
    time, and readings of one station with the same timestamp are averaged and counted. All of this
    happens before the period, the interval D and the boundaries are computed.
 2. Period: a station counts on a UTC calendar day if it has at least 12 readings on it; a day
    qualifies if at least half of the candidate stations (rounded up) count on it. Qualifying days are
    joined into runs, a gap of at most 2 non-qualifying days being bridged; the period is the run
    spanning the most days (ties: more station-days, then the earlier), from the start of its first
    day to the start of the day after its last (excluded). Readings outside it are dropped and counted.
 3. Nominal interval D: the median over candidate stations of each station's median positive interval
    between consecutive readings within the period, rounded half up to whole seconds; not evaluable if
    D < 1 s.
 4. Calendar boundaries: training = first 50% of the period's elapsed time, validation = next 20%,
    test = last 30%.
 5. Eligibility, from the training span only: the station's training readings are cut into segments
    (step 6); a station is kept if it has at least one training segment longer than 12 readings and,
    pooled over those segments, its readings occupy at least 50% of the D-wide bins they span (gaps
    between segments are not counted).
 6. Segments: each kept station's readings in each span, cut wherever the elapsed time between
    consecutive readings exceeds 10 D. P_obs keeps the readings. P_fill: bins of width D from the
    segment's first reading; a bin's value is the mean of its readings and its time their mean time
    (the number of readings per bin is kept); an empty bin, which can only lie between occupied bins,
    gets the linear interpolation in elapsed time, at its centre, of the neighbouring occupied bins.
 7. Feasibility, per preparation and independently for P_obs and P_fill: evaluable only if at least one
    station is kept, the training segments give at least 1,000 windows, and validation and test each
    have at least 1,000 prediction steps; otherwise reported as not evaluable and not run.
"""
import sys, re, json, zipfile, hashlib, pickle, math
import numpy as np, pandas as pd
LOOK_BACK = 12
N_FIELDS = 17
T_MIN, T_MAX = -50.0, 50.0
GAP_FACTOR = 10
SPLIT = (0.50, 0.70)
MIN_STEPS = 1000
MAX_BRIDGE = 2
T_WINDOW = (1104537600.0, 1293840000.0)   # 2005-01-01 to 2011-01-01 UTC, around the SensorScope deployments of 2006-2008
BACKWARD_S = 3600.0
MAX_BACKWARD_SHARE = 0.05
LISTED = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 25, 28, 29, 31, 32]   # protocol manifest

def parse(raw, st, apply_backward=True):
    t, v, bad, out_range, nf = [], [], 0, 0, {}
    out_window, backward, run_max = 0, 0, -math.inf
    for line in raw.decode("utf-8", "replace").splitlines():
        f = line.split()
        if not f: continue
        nf[len(f)] = nf.get(len(f), 0) + 1
        try:
            ok = len(f) == N_FIELDS and float(f[0]) == st
            a, b = float(f[7]), float(f[8])
        except (IndexError, ValueError):
            ok = False
        if not ok or not (math.isfinite(a) and math.isfinite(b)): bad += 1; continue
        if not (T_WINDOW[0] <= a < T_WINDOW[1]): out_window += 1; continue          # missing, wrong unit or implausible date
        if apply_backward and a < run_max - BACKWARD_S: backward += 1; continue        # clock reset: earlier than data already logged
        if not (T_MIN < b < T_MAX): out_range += 1; continue
        run_max = max(run_max, a)
        t.append(a); v.append(b)
    n_lines = sum(nf.values())
    if apply_backward and n_lines and backward > MAX_BACKWARD_SHARE * n_lines:    # the file is not in time order: rule not applied
        r = parse(raw, st, apply_backward=False); r["not_time_ordered"] = True; r["backward_jumps_if_applied"] = backward; return r
    s = pd.Series(v, dtype=float).groupby(np.array(t, dtype=float)).mean().sort_index()
    return dict(t=s.index.to_numpy(float), v=s.to_numpy(float), malformed=bad, outside_time_window=out_window,
                backward_jumps=backward, out_of_range=out_range, duplicates_merged=len(t) - len(s),
                field_counts={str(k): c for k, c in sorted(nf.items())}, not_time_ordered=False)

def load(zpath):
    z = zipfile.ZipFile(zpath); out = {}
    for n in sorted(z.namelist()):
        m = re.fullmatch(r"(?:.*/)?stbernard-meteo-(\d+)\.txt", n)
        if m: out[int(m.group(1))] = parse(z.read(n), int(m.group(1)))
    return out

def half_up(x): return float(math.floor(x + 0.5))

def period(S):
    days = {}
    for r in S.values():                                  # a station counts on a day with at least 12 readings
        d, n = np.unique(np.floor(r["t"] / 86400.0), return_counts=True)
        for x in d[n >= LOOK_BACK]: days[x] = days.get(x, 0) + 1
    need = math.ceil(len(S) / 2)
    ok = sorted(d for d, c in days.items() if c >= need)
    if not ok: return None
    runs, cur = [], [ok[0]]
    for d in ok[1:]:
        if d - cur[-1] <= MAX_BRIDGE + 1: cur.append(d)   # runs separated by at most 2 non-qualifying days are joined
        else: runs.append(cur); cur = [d]
    runs.append(cur)
    best = max(runs, key=lambda r: (r[-1] - r[0], sum(days[d] for d in r), -r[0]))
    return best[0] * 86400.0, (best[-1] + 1) * 86400.0

def span_masks(t, P0, b1, b2, P1):
    return {"train": (t >= P0) & (t < b1), "val": (t >= b1) & (t < b2), "test": (t >= b2) & (t < P1)}

def cut(t, D):
    brk = np.flatnonzero(np.diff(t) > GAP_FACTOR * D)
    return list(zip(np.r_[0, brk + 1], np.r_[brk + 1, len(t)]))

def fill_segment(tt, vv, D):
    k = np.floor((tt - tt[0]) / D).astype(int)
    g = pd.DataFrame({"k": k, "t": tt, "v": vv}).groupby("k").agg(t=("t", "mean"), v=("v", "mean"), n=("v", "size"))
    kk = np.arange(k.max() + 1); centre = tt[0] + (kk + 0.5) * D
    have = np.isin(kk, g.index.to_numpy())
    tm = np.where(have, g["t"].reindex(kk).to_numpy(), centre)
    val = g["v"].reindex(kk).to_numpy().copy()
    cnt = g["n"].reindex(kk).fillna(0).to_numpy().astype(int)
    if (~have).any(): val[~have] = np.interp(centre[~have], g["t"].to_numpy(), g["v"].to_numpy())
    return tm, val, ~have, cnt

def build(S, keep, D, bounds, prep):
    out = []
    for st in keep:
        t, v = S[st]["t"], S[st]["v"]
        for sp, mk in span_masks(t, *bounds).items():
            ts, vs = t[mk], v[mk]
            if len(ts) == 0: continue
            for a, c in cut(ts, D):
                tt, vv = ts[a:c], vs[a:c]
                if prep == "obs":
                    out.append(dict(mote=st, span=sp, bins=tt, values=vv.copy(), filled=np.zeros(len(tt), bool), count=np.ones(len(tt), int)))
                else:
                    tm, val, fl, cnt = fill_segment(tt, vv, D)
                    out.append(dict(mote=st, span=sp, bins=tm, values=val, filled=fl, count=cnt))
    return out

def coverage(segs, keep):
    c = {}
    for sp in ("train", "val", "test"):
        S = [s for s in segs if s["span"] == sp]; L = np.array([len(s["values"]) for s in S], dtype=int)
        c[sp] = dict(segments=len(S), readings=int(L.sum()), filled=int(sum(int(s["filled"].sum()) for s in S)),
                     init_reports=int(np.minimum(L, LOOK_BACK).sum()), closed_loop_steps=int(np.clip(L - LOOK_BACK, 0, None).sum()),
                     short_segments=int((L <= LOOK_BACK).sum()), median_segment_len=float(np.median(L)) if len(L) else 0.0,
                     per_station={int(st): dict(readings=int(sum(len(s["values"]) for s in S if s["mote"] == st)),
                                                closed_loop_steps=int(sum(max(len(s["values"]) - LOOK_BACK, 0) for s in S if s["mote"] == st)))
                                  for st in keep})
    c["train_windows"] = c["train"]["closed_loop_steps"]
    return c

def feasible(cov, keep):
    why = []
    if not keep: why.append("no eligible station")
    if cov["train_windows"] < MIN_STEPS: why.append("fewer than 1,000 training windows")
    for sp in ("val", "test"):
        if cov[sp]["closed_loop_steps"] < MIN_STEPS: why.append(f"fewer than 1,000 {sp} steps")
    return (not why), why

def not_evaluable(meta, reason):
    meta["evaluable"] = {"obs": False, "fill": False}; meta["not_evaluable_reason"] = {"obs": [reason], "fill": [reason]}
    json.dump(meta, open("stb_meta.json", "w"), indent=1); print("NOT EVALUABLE:", reason)

def main(zpath):
    S = load(zpath)
    meta = dict(zip_sha256=hashlib.sha256(open(zpath, "rb").read()).hexdigest(), stations_in_archive=sorted(S),
                stations_missing_from_manifest=sorted(set(S) - set(LISTED)), manifest_stations_absent=sorted(set(LISTED) - set(S)),
                parse={st: {k: r[k] for k in ("malformed", "outside_time_window", "backward_jumps", "not_time_ordered", "out_of_range", "duplicates_merged", "field_counts")} | {"readings": int(len(r["t"]))}
                       for st, r in S.items()})
    if not S: return not_evaluable(meta, "no station file")
    P = period(S)
    if P is None: return not_evaluable(meta, "no day with half of the stations reporting")
    P0, P1 = P
    for st, r in S.items():
        mk = (r["t"] >= P0) & (r["t"] < P1); meta["parse"][st]["outside_period"] = int((~mk).sum())
        r["t"], r["v"] = r["t"][mk], r["v"][mk]
        v = r["v"]; meta["parse"][st]["isolated_jumps_gt_5C"] = int(((np.abs(v[1:-1] - v[:-2]) > 5) & (np.abs(v[1:-1] - v[2:]) > 5)).sum()) if len(v) > 2 else 0
    meds = {}
    for st, r in S.items():
        d = np.diff(r["t"]); d = d[d > 0]
        meds[st] = float(np.median(d)) if len(d) else None
    vals = [x for x in meds.values() if x is not None]
    D = half_up(float(np.median(vals))) if vals else 0.0
    meta.update(period_epoch_s=[P0, P1], period_utc=[str(pd.to_datetime(x, unit="s")) for x in (P0, P1)],
                nominal_interval_s=D, station_median_intervals_s=meds)
    if D < 1: return not_evaluable(meta, "nominal interval below 1 s")
    b1 = P0 + SPLIT[0] * (P1 - P0); b2 = P0 + SPLIT[1] * (P1 - P0); bounds = (P0, b1, b2, P1)
    elig = {}
    for st, r in S.items():
        tr = r["t"][(r["t"] >= P0) & (r["t"] < b1)]
        long_runs = [(a, c) for a, c in cut(tr, D) if c - a > LOOK_BACK] if len(tr) else []
        occ_bins = sum(len(np.unique(np.floor((tr[a:c] - tr[a]) / D))) for a, c in long_runs)      # pooled within segments
        all_bins = sum(int(np.floor((tr[c - 1] - tr[a]) / D)) + 1 for a, c in long_runs)
        occ = occ_bins / all_bins if long_runs else 0.0
        elig[st] = dict(train_readings=int(len(tr)), train_segments_gt_12=len(long_runs), train_occupancy=occ,
                        kept=bool(long_runs and occ >= 0.5))
    keep = sorted(st for st in S if elig[st]["kept"])
    meta.update(gap_limit_s=GAP_FACTOR * D, boundaries_epoch_s=list(bounds),
                boundaries_utc=[str(pd.to_datetime(x, unit="s")) for x in bounds], eligibility=elig, stations_kept=keep,
                preparations={}, evaluable={}, not_evaluable_reason={})
    for prep in ("obs", "fill"):
        segs = build(S, keep, D, bounds, prep)
        cov = coverage(segs, keep); ok, why = feasible(cov, keep)
        meta["preparations"][prep] = cov; meta["evaluable"][prep] = ok; meta["not_evaluable_reason"][prep] = why
        pickle.dump(segs, open(f"stb_segments_{prep}.pkl", "wb"))
    json.dump(meta, open("stb_meta.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("stations_kept", "nominal_interval_s", "boundaries_utc", "evaluable", "not_evaluable_reason")}, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
