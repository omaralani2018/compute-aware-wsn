"""Intel Berkeley Lab temperature matrix -- TIMESTAMP-INDEXED preparation.

Source: data.txt from the Intel Lab Data release (2,313,682 lines),
        sha256 21d156880db48eb140390beffc18b017282777cdc41e30e665072544ecf0913e
        (obtained via the GitHub mirror linsea423/Intel_Lab_Data, data.zip).

WHY THIS SCRIPT EXISTS. The matrix used in all earlier work (intel_full.npy,
34,478 x 33) was reconstructed to 99.976% cell agreement by pivoting on the
EPOCH column with these same rules. That index is wrong: each mote's epoch
counter restarts on reboot (40-190 restarts per mote over the deployment), so
one epoch number spans readings taken weeks apart. On the matrix's own rows,
21.7% combine readings from motes more than a day apart and 15,924 cells are
means of readings from different days. This script applies the SAME rules on a
calendar-time grid instead, so rows are true 31-second time bins and every
column is a chronological series.

RULES (stated before the matrix was built; identical to the recovered ones
except for the index):
  R1 keep moteid 1..54 (ids above 54 are corrupt records)
  R2 validity: 0 < temperature < 50 degC; NaN dropped
  R3 time grid: 31-second bins on the timestamp, origin 2004-02-28 00:00:00,
     covering first..last valid reading; mean of readings falling in one bin
  R4 mote inclusion: <= 50% missing bins over the mote's OWN active span
     (first to last valid reading of that mote). Motes start and stop on
     different days (median active span 24 days of the 37-day deployment), and
     the recovered epoch-index rule implicitly measured missingness over each
     mote's own range; measuring it over the whole grid instead admits only 5
     motes. Both counts are recorded in the metadata.
  R5 linear interpolation of interior gaps of at most 8 consecutive bins
  R6 complete-case rows (a row is kept only if every included mote has a value)
Rows are therefore time-ordered but not uniformly spaced; the gap structure is
reported alongside the matrix and rows separated by a long gap are NOT joined
by interpolation (R5 caps at 8 bins = ~4 minutes).
"""
import pandas as pd, numpy as np, hashlib, json, sys
SRC="raw/data.txt"; BIN="31s"; ORIGIN="2004-02-28 00:00:00"
cols=["date","time","epoch","moteid","temperature","humidity","light","voltage"]
raw=pd.read_csv(SRC,sep=" ",header=None,names=cols).dropna(subset=["moteid"]); raw["moteid"]=raw.moteid.astype(int)
d=raw[(raw.moteid>=1)&(raw.moteid<=54)]                                    # R1
d=d[(d.temperature>0)&(d.temperature<50)].dropna(subset=["temperature"])  # R2
d=d.assign(ts=pd.to_datetime(d.date+" "+d.time,errors="coerce")).dropna(subset=["ts"])
d["bin"]=d.ts.dt.floor(BIN)                                                 # R3 (floor to 31 s from midnight-aligned origin)
origin=pd.Timestamp(ORIGIN); d["bin"]=origin+((d.ts-origin)//pd.Timedelta(BIN))*pd.Timedelta(BIN)
p=d.pivot_table(index="bin",columns="moteid",values="temperature",aggfunc="mean")
grid=pd.date_range(p.index.min(),p.index.max(),freq=BIN); p=p.reindex(grid)
miss_full=p.isna().mean()
own={m:p[m].loc[p[m].first_valid_index():p[m].last_valid_index()].isna().mean() for m in p.columns}
own=pd.Series(own); keep=p.columns[own<=0.5]; p=p[keep]                 # R4 (own active span)
R4_note=dict(motes_le50_missing_full_grid=int((miss_full<=0.5).sum()),motes_le50_missing_own_span=int(len(keep)))
p=p.interpolate(limit=8,limit_area="inside")                                # R5
q=p.dropna()                                                                # R6
M=q.to_numpy(); np.save("intel_time.npy",M)
gaps=np.diff(q.index.to_numpy()).astype("timedelta64[s]").astype(int)
meta=dict(R4_alternatives=R4_note,source_sha256=hashlib.sha256(open(SRC,"rb").read()).hexdigest(),bin=BIN,origin=ORIGIN,
  rows=int(M.shape[0]),motes=[int(m) for m in q.columns],grid_bins=int(len(grid)),
  first_row=str(q.index[0]),last_row=str(q.index[-1]),
  consecutive_rows_adjacent_bins=float((gaps==31).mean()),gap_gt_1h=int((gaps>3600).sum()),gap_max_h=float(gaps.max()/3600),
  matrix_sha256=hashlib.sha256(M.tobytes()).hexdigest())
tr,va=int(.5*len(q)),int(.7*len(q))
meta["split_calendar"]={"train":[str(q.index[0]),str(q.index[tr-1])],"val":[str(q.index[tr]),str(q.index[va-1])],"test":[str(q.index[va]),str(q.index[-1])]}
meta[">40C_by_span"]={k:int((M[a:b]>40).sum()) for k,(a,b) in {"train":(0,tr),"val":(tr,va),"test":(va,len(M))}.items()}
json.dump(meta,open("intel_time_meta.json","w"),indent=1); pd.Series(q.index).to_csv("intel_time_rows.csv",index=False,header=["bin_start"])
print(json.dumps(meta,indent=1))
