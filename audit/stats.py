import json, csv, numpy as np
import os; WORKDIR = os.path.abspath(os.environ.get("WORKDIR", "."))   # the directory the runs were made in (see README)
from scipy.stats import wilcoxon
C=json.load(open(f"{WORKDIR}/audit/cheap.json"))
MAC={"lightweight":208,"cnn_lstm":99488}
claimed={(r["dataset"],r["model"]):r for r in csv.DictReader(open(f"{WORKDIR}/benchmark_run/main_comparison_tau010.csv"))}
rows=[]; issues=[]
for ds,c in C.items():
    arm=max(c["arms"].items(),key=lambda kv:kv[1][0])  # first max in insertion order persistence,expsm,linear
    vals=sorted([v[0] for v in c["arms"].values()],reverse=True)
    if abs(vals[0]-vals[1])<1e-12: issues.append(f"{ds}: arm-level val tie")
    ct=arm[1][1]
    J=json.load(open(f"{WORKDIR}/learned_{ds}.json"))
    ent=J["entries"]
    for m in MAC:
        e=[r for r in ent if r["predictor"]==m]
        seeds=sorted(r["seed"] for r in e)
        if seeds!=list(range(20)): issues.append(f"{ds}/{m}: seeds {seeds}")
        if not all(r["receiver_verified"] is True for r in e): issues.append(f"{ds}/{m}: receiver not verified")
        v=np.array([r["test_rho"] for r in sorted(e,key=lambda r:r["seed"])])
        d=v-ct; p=wilcoxon(d).pvalue; dm=d.mean()
        ers=MAC[m]*1e-9/dm if dm>0 else None
        sig=p<0.05/22
        rows.append((ds,m,arm[0],ct,v.mean(),v.std(),dm,p,ers,sig,(d==0).sum()))
        cl=claimed[(ds,m)]
        diffs=[]
        if cl["cheap_selected_on_val"]!=arm[0]: diffs.append("arm")
        for key,val in [("cheap_test_rho",ct),("model_rho",v.mean()),("delta_rho",dm),("model_sd",v.std())]:
            if abs(float(cl[key])-val)>6e-5: diffs.append(f"{key} {cl[key]} vs {val:.5f}")
        if abs(float(cl["wilcoxon_p"])/p-1)>0.01: diffs.append(f"p {cl['wilcoxon_p']} vs {p:.3g}")
        if (cl["significant"]=="True")!=sig: diffs.append("sig")
        ce=cl["crossover_e_r_J"]
        if (ce=="never")!=(ers is None) or (ers and abs(float(ce)/ers-1)>0.01): diffs.append(f"e_r {ce} vs {ers}")
        if (cl["pays_at_3p8e_4"]=="True")!=(ers is not None and ers<3.8e-4): diffs.append("pays")
        if diffs: issues.append(f"{ds}/{m}: "+"; ".join(diffs))
        # other checks
        other=set(r["predictor"] for r in ent)-set(MAC)
        if other: issues.append(f"{ds}: extra predictors {other}")
    if J.get("split")!=[int(.5*c["T"]),int(.7*c["T"])] or abs(J.get("unit",0)-c["unit"])>1e-9*c["unit"] or J.get("T")!=c["T"]:
        issues.append(f"{ds}: json meta T/split/unit {J.get('T')},{J.get('split')},{J.get('unit')} vs {c['T']},{c['unit']}")
for r in rows: print("%-17s %-11s %-13s %.4f %.4f %.4f %+.4f %.3g %s %s ties=%d"%(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],"%.3g"%r[8] if r[8] else "never",r[9],r[10]))
for m in MAC:
    s=[r for r in rows if r[1]==m]
    print(m,"better",sum(r[9] and r[6]>0 for r in s),"worse",sum(r[9] and r[6]<0 for r in s),"ns",sum(not r[9] for r in s),"pays",sum(r[8] is not None and r[8]<3.8e-4 for r in s))
print("ISSUES:",*issues,sep="\n")
