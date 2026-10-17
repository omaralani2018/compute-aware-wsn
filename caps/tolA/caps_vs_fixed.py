import json, numpy as np
RUN="stage17/run"; MS=json.load(open(f"{RUN}/ms_numbers.json"))["datasets"]
ER,EOP=3.8e-4,1e-9; MAC={"lightweight":208,"cnn_lstm":99488}
DS=["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]
E=lambda rho,m:(1-rho)*ER+m*EOP
tot={}; rows=[]
for ds in DS:
    bt=MS[ds]["by_tau"]["0.1"]; sel=bt["selected"]; mR=bt["selected_macs"]; vR=bt["val"][sel]; tR=bt["test"][sel]["rho"]
    L={(r["predictor"],r["seed"]):r for r in json.load(open(f"{RUN}/learned_{ds}.json"))["entries"]}
    TA={r["seed"]:r for r in json.load(open(f"tolA/ta_{ds}.json"))["entries"]}
    out={k:[] for k in ["R","LW_mse","LW_ta","CNN","CAPS","CAPS_ta"]}
    for s in range(20):
        lw=L[("lightweight",s)]; cnn=L.get(("cnn_lstm",s)); ta=TA[s]
        out["R"].append(E(tR,mR)); out["LW_mse"].append(E(lw["test_rho"],208)); out["LW_ta"].append(E(ta["test_rho"],208))
        if cnn: out["CNN"].append(E(cnn["test_rho"],99488))
        for key,lwr in (("CAPS",lw),("CAPS_ta",ta)):
            cands=[("R",vR,tR,mR),("lw",lwr["val_rho"],lwr["test_rho"],208)]
            if cnn and not ((99488-mR)*EOP>=(1-vR)*ER): cands.append(("cnn",cnn["val_rho"],cnn["test_rho"],99488))   # Screen
            best=max(cands,key=lambda c:(c[1]-vR)*ER-(c[3]-mR)*EOP)                                          # Select
            out[key].append(E(best[2],best[3]))
    med={k:float(np.median(v)) for k,v in out.items() if v}
    rows.append((ds,med))
    for k,v in med.items(): tot[k]=tot.get(k,0)+v
print("%-17s"%"stream"+"".join("%10s"%k for k in ["R","LW_mse","LW_ta","CNN","CAPS","CAPS_ta"]))
for ds,m in rows: print("%-17s"%ds+"".join("%10.2f"%(1e5*m[k]) if k in m else "%10s"%"-" for k in ["R","LW_mse","LW_ta","CNN","CAPS","CAPS_ta"]))
print("%-17s"%"SUM (1e-5 J)"+"".join("%10.2f"%(1e5*tot[k]) for k in ["R","LW_mse","LW_ta","CNN","CAPS","CAPS_ta"]))
best_fixed=min(tot["R"],tot["LW_mse"],tot["LW_ta"],tot["CNN"])
print("CAPS_ta vs best fixed: %.1f%% less energy; vs worst fixed: %.1f%%"%(100*(1-tot["CAPS_ta"]/best_fixed),100*(1-tot["CAPS_ta"]/max(tot["R"],tot["LW_mse"],tot["LW_ta"],tot["CNN"]))))
wins=sum(1 for ds,m in rows if m["CAPS_ta"]<=min(m["R"],m["LW_mse"],m["LW_ta"])+1e-12)
print("streams where CAPS_ta <= every fixed cheap/lightweight strategy:",wins,"/ 11")
