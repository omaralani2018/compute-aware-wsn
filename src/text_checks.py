"""Numbers quoted in the text that are not tables or figures (version 1.1).
  1. Section 5.2: the anchor radio cost as the LEACH per-report energy on the 30-mote layout, and the
     per-report energy of direct transmission (from routing_30motes.json, E_INIT = 0.5 J, E_SENSE).
  2. Section 5.2: verdicts at the anchor are unchanged across 2.0e-4 .. 4.1e-4 J (per-seed Wilcoxon,
     Bonferroni over the 22 comparisons, as in ms_numbers.py), plus the first e_r below/above the anchor
     at which any verdict changes.
  3. Sections 5.3 and 6.1: open-loop fraction of first differences within tau on the Intel TEST span.
  4. Section 5.2: complete rows of the 30-mote matrix when interpolation (rule R5) is removed;
     needs the raw archive (--raw raw/data.txt), which is not redistributed.
  5. Section 3.2 (version 1.5): energy-budget lifetime of direct transmission with every node reporting
     (routing_30motes.json) and, by the same estimate with sensing unchanged, with the validation-selected
     Intel reference suppression at tau = 0.10 at every node.
Writes text_checks.json. Run from the working directory used for ms_numbers.py."""
import json, sys, numpy as np
from scipy.stats import wilcoxon
N=json.load(open("ms_numbers.json")); D=N["datasets"]; C=N["constants"]; E_OP=C["E_OP"]; E_S=C["E_SENSE"]; E_R=C["E_R"]; MAC=C["MACS"]
out={}
# 1 -------------------------------------------------------------------------
R=json.load(open("routing_30motes.json"))["summary"]; E_INIT=0.5
out["anchor"]=dict(leach_per_report_J=E_INIT/R["leach"]["net_life_mean"]-E_S, direct_per_report_J=E_INIT/R["direct"]["net_life_mean"]-E_S,
                   relay_radius_tx_J=4000*50e-9+4000*10e-12*8.06**2, relay_radius_txrx_J=2*4000*50e-9+4000*10e-12*8.06**2, anchor_used_J=E_R)
# 2 -------------------------------------------------------------------------
thr=0.05/N["bonferroni_main_n"]; rows=[(ds,m) for ds in D for m in MAC if not D[ds]["learned"].get(m,{}).get("pending")]
G={}
for ds,m in rows:
    b=D[ds]["by_tau"]["0.1"]; rr=b["test"][b["selected"]]["rho"]; rm=b["selected_macs"]
    v=np.array([r["test_rho"] for r in json.load(open(f"learned_{ds}.json"))["entries"] if r["predictor"]==m]); G[(ds,m)]=(v-rr,(MAC[m]-rm)*E_OP)
def verdicts(er):
    o={}
    for k,(dd,dE) in G.items():
        g=er*dd-dE; o[k]=("saves" if g.mean()>0 else "costs") if wilcoxon(g)[1]<thr else "n.s."
    return o
base=verdicts(E_R); grid=np.round(np.arange(1.0e-4,8.0e-4+1e-12,0.1e-4),6)
changes={f"{er:.1e}":[f"{k[0]}/{k[1]}" for k in base if verdicts(er)[k]!=base[k]] for er in grid}
below=[float(er) for er in grid if er<E_R and changes[f"{er:.1e}"]]; above=[float(er) for er in grid if er>E_R and changes[f"{er:.1e}"]]
out["verdict_band"]=dict(n_comparisons=len(rows),bonferroni_threshold=thr,
    unchanged_2p0e4_to_4p1e4=all(not changes[f"{er:.1e}"] for er in grid if 2.0e-4-1e-12<=er<=4.1e-4+1e-12),
    highest_er_below_anchor_with_change=max(below) if below else None, lowest_er_above_anchor_with_change=min(above) if above else None,
    changes_by_er={k:v for k,v in changes.items() if v})
# 3 -------------------------------------------------------------------------
M=np.load("intel_time.npy"); T=len(M); tr,va=int(C["SPLIT"][0]*T),int(C["SPLIT"][1]*T); d=np.abs(np.diff(M[va:],axis=0))
out["open_loop_test_span"]={str(t):float((d<=t).mean()) for t in (0.05,0.10,0.20,0.50)}
# 4 -------------------------------------------------------------------------
raw=sys.argv[sys.argv.index("--raw")+1] if "--raw" in sys.argv else None
if raw:
    import pandas as pd
    cols=["date","time","epoch","moteid","temperature","humidity","light","voltage"]
    r=pd.read_csv(raw,sep=" ",header=None,names=cols).dropna(subset=["moteid"]); r["moteid"]=r.moteid.astype(int)
    r=r[(r.moteid>=1)&(r.moteid<=54)]; r=r[(r.temperature>0)&(r.temperature<50)].dropna(subset=["temperature"])
    r=r.assign(ts=pd.to_datetime(r.date+" "+r.time,errors="coerce")).dropna(subset=["ts"])
    o=pd.Timestamp("2004-02-28 00:00:00"); r["bin"]=o+((r.ts-o)//pd.Timedelta("31s"))*pd.Timedelta("31s")
    p=r.pivot_table(index="bin",columns="moteid",values="temperature",aggfunc="mean")
    motes=json.load(open("intel_time_meta.json"))["motes"]
    out["stress_test_no_interpolation"]=dict(motes=len(motes),complete_rows_without_R5=int(p[motes].dropna().shape[0]),complete_rows_with_R5=int(T))
# 5 -------------------------------------------------------------------------
b=D["intel_time"]["by_tau"]["0.1"]; rho=b["test"][b["selected"]]["rho"]; e_d=out["anchor"]["direct_per_report_J"]
out["lifetime_section_3_2"]=dict(direct_every_report_rounds=R["direct"]["net_life_mean"],reference_suppression=rho,
    direct_with_reference_suppression_rounds=E_INIT/(E_S+(1-rho)*e_d))
json.dump(out,open("text_checks.json","w"),indent=1); print(json.dumps(out,indent=1))
