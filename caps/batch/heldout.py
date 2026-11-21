import sys, os, json, importlib.util, numpy as np
B=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,B)
import batch as bt   # chdir to gate/run, ol, fb
ol,fb=bt.ol,bt.fb
C=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
def mod(p):
    s=importlib.util.spec_from_file_location(os.path.basename(os.path.dirname(p)),p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cA,cC=mod(C+'/confA/conf_data.py'),mod(C+'/confC/conf_data.py')
def dps(M,f,u): return fb.dps(M,f,tau=0.1,look_back=12,train_frac=1e-6,unit=u)["rho"]
res={}
for src,names in ((cA,list(cA.FILES)),(cC,list(cC.FILES))):
    for ds in names:
        A,u=src.load(ds); A=np.asarray(A,float); T=len(A); tr,va=int(.5*T),int(.7*T); VAL,TEST=A[tr-12:va],A[va-12:]
        a=max(fb.ALPHA_GRID,key=lambda a:(dps(VAL,fb.make_expsm(a),u),-a)); k=max(fb.LIN_K_GRID,key=lambda k:(dps(VAL,fb.make_ar(k),u),-k))
        arms={"persistence":None,"expsm":fb.make_expsm(a),"linear_extrap":fb.make_ar(k)}
        val={n:dps(VAL,f,u) for n,f in arms.items()}; sel=max(val,key=val.get)
        r=bt.analyse(ds,TEST,arms[sel],u); r['ref']=sel; res[ds]=r
        m,nr=r['M3'],r['nRF52840']
        print('%-20s N=%d rho=%.3f | M3 S0=%.2e S1=%.2e S2=%.2e | nRF S0=%.2e S1=%.2e S2=%.2e'%(ds,A.shape[1],r['rho'],m['S0'],m['S1']['25'],m['S2']['25'],nr['S0'],nr['S1']['25'],nr['S2']['25']),flush=True)
json.dump(res,open(B+'/heldout_batch.json','w'),indent=1)
