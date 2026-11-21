import sys, os, json, statistics as st, numpy as np
B=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,B)
import batch as bt; ol,fb=bt.ol,bt.fb
SC=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
PLAT={"m3_float":("iotlab/iotlab_m3_summary.json","M3"),"m3_int":("iotlab/fixed/fixed_m3_21_accuracy_q_summary.json","M3"),"nrf":("iotlab/iotlab_nrf_summary.json","nRF52840")}
Mp={"expsm_q":"expsm","lin2_q":"lin2","lin12_q":"lin12","light_q":"lightweight","cnn_q":"cnn_lstm"}
E={p:{Mp.get(k,k):v["E_inc_J"] for k,v in json.load(open(os.path.join(SC,f)))["per_forecast"].items()} for p,(f,_) in PLAT.items()}
def eref(p,sel,K):
    e=E[p]; return {"persistence":e["persistence"],"expsm":e["expsm"],"linear_extrap":e["lin2"]+(K-2)/10*(e["lin12"]-e["lin2"])}[sel]
def s2(F,node,D=25):
    return sum(bt.cost(node,b) for j in range(F.shape[1]) for b in bt.s2_frames(F[:,j],D))
DS=["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]
res={}
for ds in DS:
    A,unit=ol.load(ds); A=np.asarray(A,float); A=A[:,None] if A.ndim==1 else A
    T=A.shape[0]; va=int(fb.SPLIT[1]*T); X=A[va-ol.LB:]
    sel,fR,_,_,_=ol.ref_arm(ds); K=ol.MS[ds]['by_tau']['0.1'].get('K',2)
    FR=bt.flags(X,fR,unit); n=FR.size
    rad={nd:{'S0':FR.sum()*bt.cost(nd,4)/n,'S2':s2(FR,nd)/n} for nd in ('M3','nRF52840')}
    seeds=[s for s in range(20) if os.path.exists(f'weights/{ds}_lightweight_s{s}.npz')]
    rows=[]
    for s in seeds:
        fL=ol.learned_arm(A,'lightweight',f'weights/{ds}_lightweight_s{s}.npz'); FL=bt.flags(X,fL,unit)
        rows.append({nd:{'S0':FL.sum()*bt.cost(nd,4)/n,'S2':s2(FL,nd)/n} for nd in ('M3','nRF52840')})
    out={}
    for p,(_,nd) in PLAT.items():
        dE=E[p]['lightweight']-eref(p,sel,K)
        for sc in ('S0','S2'):
            net=[rad[nd][sc]-r[nd][sc]-dE for r in rows]
            out[f'{p}/{sc}']=st.median(net)
    res[ds]=dict(seeds=len(seeds),net=out)
    print('%-17s seeds=%d '%(ds,len(seeds))+' '.join('%s=%+.2e'%(k,v) for k,v in out.items()),flush=True)
json.dump(res,open(B+'/c3.json','w'),indent=1)
for p in PLAT:
    for sc in ('S0','S2'):
        pos=[d for d in DS if res[d]['net'][f'{p}/{sc}']>0]; print(p,sc,'lightweight positive on %d/11'%len(pos),pos)
