"""RULE6: NLMS dual predictor initialised at the zero-order hold (Meghana and Parameshachari, AIKIIE 2026)
under the protocol of matched_baselines.py. See RULE6.txt."""
import sys, os, json, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
G=os.environ.get('CAW_RUN',os.path.abspath(os.path.join(HERE,'../../run'))); sys.path.insert(0,G); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../../src'))); sys.path.insert(0,os.path.abspath(os.path.join(HERE,'../batch')))
import matched_baselines as mb
cwd=os.getcwd(); os.chdir(G); import oracle_lib as ol; os.chdir(cwd)
P_GRID=[2,4,8,12]; MU_GRID=[0.01,0.03,0.06,0.1,0.3,1.0]
def macs(p): return 3*p+1
class NLMSZOH:
    def __init__(self, n, p, mu, eps, warm):
        W=np.array(warm,float); self.p,self.mu,self.eps=p,mu,eps
        self.w=np.zeros((n,p)); self.w[:,0]=1.0                      # w(0)=[1,0,...,0], newest first
        self.hist=W[-p:][::-1].T.copy()                               # (n,p) newest first
    def predict(self): return (self.w*self.hist).sum(1)
    def update(self, sent, value, pred):
        e=value-pred                                                  # zero on suppressed rounds
        nrm=(self.hist**2).sum(1)+self.eps
        self.w=self.w+self.mu*(e/nrm)[:,None]*self.hist
        self.hist=np.concatenate([value[:,None],self.hist[:,:-1]],1)
def run(ds):
    A,unit=ol.load(ds); A=np.asarray(A,float); A=A[:,None] if A.ndim==1 else A
    T=A.shape[0]; tr,va=int(mb.SPLIT[0]*T),int(mb.SPLIT[1]*T)
    VAL,TEST,TRAIN=A[tr-mb.LOOK_BACK:va],A[va-mb.LOOK_BACK:],A[:tr]
    ex=float((TRAIN**2).mean())
    mk=lambda p,mu:(lambda warm: NLMSZOH(A.shape[1],p,mu,1e-9*p*ex,warm))
    # sanity: persistence through the same harness reproduces the reference when the reference is persistence
    pers=mb.evaluate(TEST,lambda w: mb.Persistence(A.shape[1],w),"ours",1,unit)["rho"]
    pub=mb.evaluate(TEST,mk(4,0.06),"ours",1,unit,verify=True)
    best=None; grid=[]
    for p in P_GRID:
        for mu in MU_GRID:
            v=mb.evaluate(VAL,mk(p,mu),"ours",1,unit)["rho"]; grid.append([p,mu,v])
            if best is None or v>best[0]: best=(v,p,mu)
    v,p,mu=best; sel=mb.evaluate(TEST,mk(p,mu),"ours",1,unit,verify=True)
    sel_name,_,ref_macs,_,ref_rho=ol.ref_arm(ds)
    g=lambda rho,m: mb.E_R*(rho-ref_rho)-(m-ref_macs)*mb.E_OP
    return dict(ds=ds,unit=unit,ref=sel_name,ref_rho=ref_rho,ref_macs=ref_macs,persistence_test=pers,
                published=dict(p=4,mu=0.06,macs=macs(4),test=pub,g_J=g(pub["rho"],macs(4))),
                selected=dict(p=p,mu=mu,val_rho=v,macs=macs(p),test=sel,g_J=g(sel["rho"],macs(p))),val_grid=grid)
if __name__=="__main__":
    DS=["intel_time","taxi","traffic","cpu","machine_temp","grok_asg_anomaly","rds_cpu_util","iio_networkin","twitter_aapl","exchange_cpc","occupancy"]
    out={}
    for ds in (sys.argv[1:] or DS):
        r=run(ds); out[ds]=r
        print("%-17s ref=%-13s %.3f | pub %.3f | sel p=%d mu=%.2f %.3f | g_sel %+.2e | recv %s/%s"%(ds,r['ref'],r['ref_rho'],r['published']['test']['rho'],r['selected']['p'],r['selected']['mu'],r['selected']['test']['rho'],r['selected']['g_J'],r['selected']['test']['receiver_identical'],r['selected']['test']['bound_holds']),flush=True)
    json.dump(out,open(os.path.join(HERE,'nlms_results.json'),'w'),indent=1,default=float)
