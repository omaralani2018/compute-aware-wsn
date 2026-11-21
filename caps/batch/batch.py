import sys, os, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); G=os.environ.get('CAW_RUN',os.path.abspath(os.path.join(HERE,'../../run'))); sys.path.insert(0,HERE); sys.path.insert(0,G); os.chdir(G)
import oracle_lib as ol; fb=ol.fb; LB,TAU=ol.LB,ol.TAU
PTS={'M3':(2.672594889006969e-4,6.16171610040477e-4),'nRF52840':(2.655994878697281e-5,1.0807910920068709e-4)}
def cost(node,b):
    s,l=PTS[node]; assert 1<=b<=100; return s+(l-s)*(b-4)/96.0
def flags(X,f,unit):
    T,N=X.shape; R=X.copy(); F=np.zeros((T-LB,N),bool)
    for t in range(LB,T):
        p=ol.predict(f,R,t); sup=np.abs(X[t]-p)/unit<=TAU; R[t]=np.where(sup,p,X[t]); F[t-LB]=~sup
    return F
def s2_frames(col,D,cap=20):
    """col: bool report flags of one node. returns list of payload sizes (5 B per report)."""
    fr=[];buf=[]
    for t,r in enumerate(col):
        if r: buf.append(t)
        if buf and (len(buf)>=cap or t-buf[0]>=D-1): fr.append(5*len(buf)); buf=[]
    if buf: fr.append(5*len(buf))
    return fr
def analyse(ds,X,f,unit):
    F=flags(X,f,unit); n,N=F.shape; out={'rho':float(1-F.mean())}
    for node in PTS:
        e0=F.sum()*cost(node,4)/(n*N)
        e1={B:N*np.ceil(n/B)*cost(node,4*B)/(n*N) for B in (5,10,25)}
        e2={}
        for D in (5,10,25):
            tot=sum(cost(node,b) for j in range(N) for b in s2_frames(F[:,j],D)); e2[D]=tot/(n*N)
        out[node]=dict(S0=e0,S1={str(k):v for k,v in e1.items()},S2={str(k):v for k,v in e2.items()})
    return out
if __name__=='__main__':
    res={}
    for ds in ["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]:
        A,unit=ol.load(ds); A=np.asarray(A,float); A=A[:,None] if A.ndim==1 else A
        T=A.shape[0]; va=int(fb.SPLIT[1]*T); X=A[va-LB:]
        name,f,_,_,rt=ol.ref_arm(ds); r=analyse(ds,X,f,unit); assert abs(r['rho']-rt)<1e-9; r['ref']=name; res[ds]=r
        m,nr=r['M3'],r['nRF52840']
        print('%-17s rho=%.3f | M3 S0=%.2e S1(25)=%.2e S2(25)=%.2e | nRF S0=%.2e S1=%.2e S2=%.2e'%(ds,r['rho'],m['S0'],m['S1']['25'],m['S2']['25'],nr['S0'],nr['S1']['25'],nr['S2']['25']),flush=True)
    json.dump(res,open(os.path.join(HERE,'dev_batch.json'),'w'),indent=1)
