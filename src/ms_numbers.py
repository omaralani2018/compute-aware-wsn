"""Every number the rebuilt manuscript uses, from the frozen protocol's outputs.
Writes ms_numbers.json. Intel = intel_time (timestamp-indexed) when its learned
runs exist; otherwise Intel learned entries are marked pending."""
import json, os, glob, collections, numpy as np, frozen_benchmark as fb, nab_data as nd
from scipy.stats import wilcoxon
TAUS=[0.05,0.10,0.20,0.50]; E_R=fb.E_R; E_OP=fb.E_OP; E_S=fb.E_SENSE; MAC={"lightweight":208,"cnn_lstm":99488}
ER_GRID=[1e-4,3.75e-4,3.8e-4,1e-3,3e-3]; EOP_GRID=[0.5e-9,1e-9,2e-9]
def spans(A):
    A=np.asarray(A,float); A=A[:,None] if A.ndim==1 else A; T=len(A); tr,va=int(.5*T),int(.7*T)
    return A[:tr],A[tr-fb.LOOK_BACK:va],A[va-fb.LOOK_BACK:],tr,va
def dps(M,f,u,t): return fb.dps(M,f,tau=t,look_back=fb.LOOK_BACK,train_frac=1e-6,unit=u)
def cheap_select(VAL,TEST,u,t):
    a=max(fb.ALPHA_GRID,key=lambda a:(dps(VAL,fb.make_expsm(a),u,t)["rho"],-a))
    k=max(fb.LIN_K_GRID,key=lambda k:(dps(VAL,fb.make_ar(k),u,t)["rho"],-k))
    arms={"persistence":(None,0),"expsm":(fb.make_expsm(a),12),"linear_extrap":(fb.make_ar(k),k)}
    val={n:dps(VAL,f,u,t)["rho"] for n,(f,_) in arms.items()}
    sel=max(val,key=val.get)                                   # ties: dict order = persistence first
    test={n:dps(TEST,f,u,t) for n,(f,_) in arms.items()}
    return dict(alpha=a,K=k,val=val,test={n:{kk:test[n][kk] for kk in("rho","mean_err","max_err","rmse")} for n in arms},
                selected=sel,selected_macs=arms[sel][1])
def Lfull(rho_ref,mac_ref,rho,mac,er,eop): return (E_S+mac_ref*eop+(1-rho_ref)*er)/(E_S+mac*eop+(1-rho)*er)
out={"constants":dict(E_R=E_R,E_OP=E_OP,E_SENSE=E_S,LOOK_BACK=fb.LOOK_BACK,SPLIT=fb.SPLIT,EPOCH_CAP=fb.EPOCH_CAP,CKPT_EVERY=fb.CKPT_EVERY,SEEDS=len(fb.SEEDS),
     MAX_WINDOWS=fb.MAX_WINDOWS,ALPHA_GRID=fb.ALPHA_GRID,LIN_K_GRID=fb.LIN_K_GRID,MACS=MAC),"datasets":{}}
data={"intel_time":(np.load("intel_time.npy"),1.0)}
for k in nd.NAB: data[k]=nd.load(k)
for ds,(A,u) in data.items():
    TR,VAL,TEST,tr,va=spans(A); D={"T":int(len(A)),"N":int(A.shape[1]),"unit":u,"split":[tr,va],"by_tau":{}}
    if ds!="intel_time":
        D["fastness"]=nd.fastness(A); D["file"]=nd.NAB[ds].replace("nab_repo/data/","")
        v=A[:int(.5*len(A)),0]; D["fastness_train"]=float(np.std(np.diff(v))/np.std(v))
    d=np.abs(np.diff(TR,axis=0))
    for t in TAUS:
        cs=cheap_select(VAL,TEST,u,t); cs["open_loop_frac_train"]=float((d<=t*u).mean())
        D["by_tau"][str(t)]=cs
    # learned arms
    lf=f"learned_{ds}.json"; tf=f"tau_{ds}.json"; D["learned"]={}
    if os.path.exists(lf):
        ent=json.load(open(lf))["entries"]
        for m in MAC:
            e=[r for r in ent if r["predictor"]==m]
            if len(e)<20: D["learned"][m]={"pending":True,"n":len(e)}; continue
            v=np.array([r["test_rho"] for r in e]); ref=D["by_tau"]["0.1"]; rr=ref["test"][ref["selected"]]["rho"]; rm=ref["selected_macs"]
            dd=v-rr; dE=(MAC[m]-rm)*E_OP; g=E_R*dd-dE
            pos=dd[dd>0]; band=[float(dE/pos.max()),float(dE/pos.min())] if len(pos)==20 else None
            D["learned"][m]=dict(n=20,rho_mean=float(v.mean()),rho_sd=float(v.std()),max_err_mean=float(np.mean([r["test_max_err"] for r in e])),
                receiver_verified=all(r["receiver_verified"] for r in e),delta=float(dd.mean()),p_delta=float(wilcoxon(dd)[1]),
                net_saving=float(g.mean()),p_net=float(wilcoxon(g)[1]),crossover=(float(dE/dd.mean()) if dd.mean()>0 else None),crossover_range=band,
                vs_persistence_delta=float(v.mean()-ref["test"]["persistence"]["rho"]),
                L_vs_ref={ "er_grid":{str(er):Lfull(rr,rm,v.mean(),MAC[m],er,E_OP) for er in ER_GRID},
                           "eop_grid":{str(eop):Lfull(rr,rm,v.mean(),MAC[m],E_R,eop) for eop in EOP_GRID}})
    else: D["learned"]={m:{"pending":True,"n":0} for m in MAC}
    if os.path.exists(tf):
        L=json.load(open(tf))
        if len(L)==20:
            D["tau_sweep_lightweight"]={}
            for t in TAUS:
                ref=D["by_tau"][str(t)]; rr=ref["test"][ref["selected"]]["rho"]; lw=np.array([r[str(t)]["test"] for r in L]); dd=lw-rr
                D["tau_sweep_lightweight"][str(t)]=dict(rho_mean=float(lw.mean()),rho_sd=float(lw.std()),delta=float(dd.mean()),
                    p=float(wilcoxon(dd)[1]) if np.any(dd!=0) else 1.0,verified=all(r[str(t)]["verified"] for r in L))
    out["datasets"][ds]=D
# Bonferroni families
rows=[(ds,m) for ds,D in out["datasets"].items() for m in MAC if not D["learned"].get(m,{}).get("pending")]
out["bonferroni_main_n"]=len(rows); thr=0.05/len(rows) if rows else None
for ds,m in rows:
    e=out["datasets"][ds]["learned"][m]; e["sig_delta"]=e["p_delta"]<thr; e["sig_net"]=e["p_net"]<thr
sw=[(ds,t) for ds,D in out["datasets"].items() if "tau_sweep_lightweight" in D for t in D["tau_sweep_lightweight"]]
out["bonferroni_sweep_n"]=len(sw); thr2=0.05/len(sw) if sw else None
for ds,t in sw: out["datasets"][ds]["tau_sweep_lightweight"][t]["sig"]=out["datasets"][ds]["tau_sweep_lightweight"][t]["p"]<thr2
# stress test (computed once by hand from the raw archive; see prepare_intel.py rules with R5 removed)
out["intel_stress_test"]=dict(rows_no_interp=3,rows_with_interp=25127,note="with the 8-bin interpolation cap removed, only 3 bins have all 30 motes reporting")
from scipy.stats import spearmanr
xs=[];ys=[];fs=[]
for ds,D in out["datasets"].items():
    b=D["by_tau"]["0.1"]; xs.append(b["open_loop_frac_train"]); ys.append(b["test"]["persistence"]["rho"]); fs.append(D.get("fastness",np.nan))
xs=np.array(xs);ys=np.array(ys);fs=np.array(fs); m=~np.isnan(fs)
ft=np.array([D.get("fastness_train",np.nan) for D in out["datasets"].values()]); mt=~np.isnan(ft)
out["screening"]=dict(spearman_fastness_train_n10=float(spearmanr(ft[mt],ys[mt])[0]),n=int(len(xs)),spearman_openloop_frac=float(spearmanr(xs,ys)[0]),mae_openloop_frac=float(np.mean(np.abs(xs-ys))),max_abs_err=float(np.max(np.abs(xs-ys))),
                      spearman_fastness_n10=float(spearmanr(fs[m],ys[m])[0]),spearman_openloop_frac_n10=float(spearmanr(xs[m],ys[m])[0]))
json.dump(out,open("ms_numbers.json","w"),indent=1)
# summary
for ds,D in out["datasets"].items():
    r=D["by_tau"]["0.1"]; s=" | ".join("%s %.3f%s"%(m[:5],D["learned"][m]["rho_mean"],"*" if D["learned"][m].get("sig_net") else "") if not D["learned"][m].get("pending") else "%s pend(%d)"%(m[:5],D["learned"][m]["n"]) for m in MAC)
    print("%-16s ref=%-13s %.3f (P %.3f) | %s"%(ds,r["selected"],r["test"][r["selected"]]["rho"],r["test"]["persistence"]["rho"],s))
print("main comparisons:",out["bonferroni_main_n"],"sweep comparisons:",out["bonferroni_sweep_n"])
