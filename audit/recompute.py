import sys, json, csv, numpy as np
import os; WORKDIR = os.path.abspath(os.environ.get("WORKDIR", "."))   # the directory the runs were made in (see README)
sys.path.insert(0, WORKDIR)
os.chdir(WORKDIR)
from dual_prediction_fix import dual_prediction_suppress as dps
from frozen_benchmark import make_expsm, make_ar, ALPHA_GRID, LIN_K_GRID
import nab_data as nd
from scipy.stats import wilcoxon
LB=12; TAU=0.10
def ev(X,f,unit): return dps(X,f,tau=TAU,look_back=LB,train_frac=1e-6,unit=unit)["rho"]
out={}
for ds in ["intel"]+list(nd.NAB):
    if ds=="intel": A,unit=np.load("intel_full.npy"),1.0
    else: A,unit=nd.load(ds)
    A=np.asarray(A,float); A=A[:,None] if A.ndim==1 else A
    T=len(A); tr,va=int(.5*T),int(.7*T)
    VAL,TEST=A[tr-12:va],A[va-12:]
    arms={}
    arms["persistence"]=(ev(VAL,None,unit),ev(TEST,None,unit),None)
    best=None
    for a in ALPHA_GRID:
        r=ev(VAL,make_expsm(a),unit)
        if best is None or r>best[0]: best=(r,a)
    arms["expsm"]=(best[0],ev(TEST,make_expsm(best[1]),unit),best[1])
    best=None
    for k in LIN_K_GRID:
        r=ev(VAL,make_ar(k),unit)
        if best is None or r>best[0]: best=(r,k)
    arms["linear_extrap"]=(best[0],ev(TEST,make_ar(best[1]),unit),best[1])
    ranked=sorted(arms.items(),key=lambda kv:-kv[1][0])
    out[ds]=dict(arms={k:list(v) for k,v in arms.items()},T=T,unit=unit)
    print(ds,T,unit,{k:(round(v[0],5),round(v[1],5),v[2]) for k,v in arms.items()},flush=True)
json.dump(out,open(f"{WORKDIR}/audit/cheap.json","w"),indent=1)
