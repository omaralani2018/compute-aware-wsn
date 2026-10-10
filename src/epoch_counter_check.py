"""Why the Intel matrix is indexed by timestamp and not by the per-mote epoch counter (Section 5.2).
Reads the raw archive exactly as prepare_intel.py does (rules R1 and R2) and reports, for the 30 motes of
the paper and for the 33 motes of the withdrawn epoch-indexed matrix:
  - per-mote backward jumps of the epoch counter by more than 1,000 counts, taken in timestamp order, and
    how many of them are shared: at least 5 motes (the jumping mote included) jump within 30 minutes of
    the jump; the shared events are listed, and the fraction is also given without the jumps from the
    16-bit counter maximum (65,535), which all fall in one event;
  - the time spread of the readings that share one epoch number;
  - the withdrawn matrix rebuilt by pivoting on the epoch (same rules, R5 on the epoch index), its
    agreement with inputs/intel_full.npy, and how far apart in time the readings in one row are.
The jump counts are also given without the temperature filter R2, since the counter itself is not
affected by the validity of the temperature reading.
Inputs: raw/data.txt, intel_time_meta.json, old33_motes.json, intel_full.npy. Output: epoch_counter.json."""
import json, numpy as np, pandas as pd
SRC = "raw/data.txt"; JUMP = 1000; SHARED_MOTES = 5; WINDOW = pd.Timedelta("30min")
cols = ["date","time","epoch","moteid","temperature","humidity","light","voltage"]
raw = pd.read_csv(SRC, sep=" ", header=None, names=cols).dropna(subset=["moteid"]); raw["moteid"] = raw.moteid.astype(int)
raw = raw[(raw.moteid >= 1) & (raw.moteid <= 54)]                                        # R1
raw = raw.assign(ts=pd.to_datetime(raw.date+" "+raw.time, errors="coerce")).dropna(subset=["ts","epoch"])
valid = raw[(raw.temperature > 0) & (raw.temperature < 50)].dropna(subset=["temperature"])  # R2
m30 = json.load(open("intel_time_meta.json"))["motes"]; m33 = json.load(open("old33_motes.json"))

def jumps(d, motes):
    J = []
    for m, x in d[d.moteid.isin(motes)].groupby("moteid"):
        x = x.sort_values("ts", kind="mergesort"); e = x.epoch.to_numpy(); t = x.ts.to_numpy()
        for i in np.where(np.diff(e) < -JUMP)[0]: J.append((int(m), pd.Timestamp(t[i+1]), int(e[i]), int(e[i+1])))
    J = pd.DataFrame(J, columns=["mote","t","before","after"])
    per = J.groupby("mote").size().reindex(motes, fill_value=0)
    shared = np.array([J[(abs(J.t-r.t) <= WINDOW)].mote.nunique() >= SHARED_MOTES for r in J.itertuples()])
    S = J[shared].sort_values("t", kind="mergesort"); events, cur = [], []
    for r in S.itertuples():
        if cur and r.t - cur[-1].t > WINDOW: events.append(cur); cur = []
        cur.append(r)
    if cur: events.append(cur)
    events = [dict(start=str(e[0].t.floor("min")), motes=len({x.mote for x in e})) for e in events]
    notmax = (J.before < 65500).to_numpy()
    return dict(n_jumps=int(len(J)), per_mote_min=int(per.min()), per_mote_max=int(per.max()),
                motes_without_jump=int((per == 0).sum()), smallest_jump=int((J.before-J.after).min()),
                shared_fraction=float(np.mean(shared)), shared_events=events,
                shared_fraction_excluding_jumps_from_counter_maximum=float(np.mean(shared[notmax])))

def spread(d, motes):
    x = d[d.moteid.isin(motes)]; g = x.groupby("epoch").ts.agg(["min","max"])
    s = (g["max"]-g["min"])/pd.Timedelta(days=1); big = s[s > 1]
    return dict(epoch_values=int(len(s)), fraction_spanning_over_1_day=float((s > 1).mean()),
                min_span_days_among_those=float(big.min()), max_span_days=float(s.max()))

def withdrawn_matrix(d):
    x = d[d.moteid.isin(m33)]
    p = x.pivot_table(index="epoch", columns="moteid", values="temperature", aggfunc="mean")
    p = p.reindex(range(int(p.index.min()), int(p.index.max())+1)).interpolate(limit=8, limit_area="inside").dropna()
    F = np.load("intel_full.npy")
    agree = float(np.mean(np.isclose(p.to_numpy(), F, atol=1e-6))) if p.shape == F.shape else None
    c = x.groupby(["epoch","moteid"]).ts.median().reset_index()
    g = c.groupby("epoch").ts.agg(["min","max"]).reindex(p.index); s = (g["max"]-g["min"])/pd.Timedelta(days=1)
    return dict(rows=int(len(p)), shape_matches_intel_full=p.shape == F.shape, cell_agreement_with_intel_full=agree,
                rows_spanning_over_1_day=float((s > 1).mean()), rows_spanning_over_7_days=float((s > 7).mean()),
                n_rows_spanning_over_14_days=int((s > 14).sum()), max_row_span_days=float(s.max()))

out = dict(rule=dict(jump_counts=JUMP, shared_if_motes_within_30_min=SHARED_MOTES),
           jumps={f"{lab}_{fl}": jumps(d, ms) for lab, ms in (("30_motes", m30), ("33_motes", m33))
                  for fl, d in (("valid_readings", valid), ("all_readings", raw))},
           epoch_spread={lab: spread(valid, ms) for lab, ms in (("30_motes", m30), ("33_motes", m33))},
           withdrawn_epoch_matrix=withdrawn_matrix(valid))
json.dump(out, open("epoch_counter.json", "w"), indent=1); print(json.dumps(out, indent=1))
