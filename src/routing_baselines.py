"""Routing and clustering baselines of Section 6.5 / Table 5.

The simulator is extracted from the original analysis notebook (WSN_v5_3_CNN_FGA, sections 1, 4 and 5:
first-order radio model, connectivity graph, Dijkstra, fuzzy-genetic routing, random multi-hop routing,
LEACH with a probability sweep and 50 cluster draws) and made standalone. It is topology-only: every node
reports every round (emit = 1) and no compute is charged, so only the routing lever acts.

Network lifetime = N * E_INIT / sum_i E_i  (rounds), first node death = E_INIT / max_i E_i.
The sink is placed outside the field corner (pos.min - 0.15 * span); the relay radius is 1.9 x the median
nearest-neighbour spacing (8.06 m on the Intel layout); every node may also reach the sink directly.

Usage:
  python3 routing_baselines.py                       # 30-mote layout of intel_time_meta.json, seeds 0..19
  python3 routing_baselines.py --motes old33.json --seeds 0,1,2,3,4
Outputs routing_<n>motes.json (per-seed results and a summary).
Inputs: mote_locs.txt (Intel Lab release), intel_time_meta.json (written by prepare_intel.py)."""
import math, heapq, sys, json
import numpy as np, pandas as pd

E_ELEC=50e-9; EPS_FS=10e-12; EPS_MP=0.0013e-12; PKT_BITS=4000; E_SENSE=1e-7; E_INIT=0.5
D0=math.sqrt(EPS_FS/EPS_MP); GA_POP=20; GA_GEN=20; GA_PCROSS=0.9; GA_PMUT=0.05

def run(pos, seed=0):
    rng=np.random.default_rng(seed); N=len(pos)
    def _nn(P):
        d=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1)); np.fill_diagonal(d,np.inf); return d.min(1)
    comm_r=float(1.9*np.median(_nn(pos))); span=float(np.linalg.norm(pos.max(0)-pos.min(0)))
    SINKXY=pos.min(0)-0.15*span; Pg=np.vstack([pos,SINKXY]); SINK=N
    hyp=lambda a,b: float(np.hypot(Pg[a,0]-Pg[b,0],Pg[a,1]-Pg[b,1]))
    etx=lambda bits,d: bits*E_ELEC+(bits*EPS_FS*d*d if d<D0 else bits*EPS_MP*(d**4))
    erx=lambda bits: bits*E_ELEC
    M=N+1; A=[[j for j in range(M) if j!=i and hyp(i,j)<=comm_r] for i in range(M)]
    def reach():
        seen=[False]*M; st=[SINK]; seen[SINK]=True
        while st:
            u=st.pop()
            for v in A[u]:
                if not seen[v]: seen[v]=True; st.append(v)
        return seen
    seen=reach(); g=0
    while not all(seen) and g<M+2:
        g+=1; R=[k for k in range(M) if seen[k]]
        for u in range(M):
            if not seen[u]:
                k=min(R,key=lambda k:hyp(u,k)); A[u].append(k); A[k].append(u)
        seen=reach()
    def dijkstra(src,dst):
        INF=1e18; d=[INF]*M; prev=[-1]*M; d[src]=0.0; pq=[(0.0,src)]
        while pq:
            du,u=heapq.heappop(pq)
            if du>d[u]: continue
            if u==dst: break
            for v in A[u]:
                w=etx(PKT_BITS,hyp(u,v))+erx(PKT_BITS)
                if d[u]+w<d[v]: d[v]=d[u]+w; prev[v]=u; heapq.heappush(pq,(d[v],v))
        if d[dst]>=INF: return None
        p=[]; u=dst
        while u!=-1: p.append(u); u=prev[u]
        return p[::-1]
    linkq=lambda a,b: float(np.clip(1.0-hyp(a,b)/max(comm_r,1e-9),0.0,1.0))
    def tri(x,a,b,c):
        if x<=a or x>=c: return 0.0
        return (x-a)/(b-a) if x<b else (c-x)/(c-b)
    def fuzzy_reward(res_e,buf,lq,dn):
        e_hi=tri(res_e,0.2,1.0,1.0); e_lo=tri(res_e,0.0,0.0,0.6); b_hi=tri(buf,0.2,1.0,1.0)
        q_hi=tri(lq,0.2,1.0,1.0); d_sh=tri(dn,0.0,0.0,0.8)
        return (0.30*e_hi+0.15*b_hi+0.25*q_hi+0.30*d_sh)*(1.0-0.3*e_lo)
    node_attr=lambda i,en: en[i]/E_INIT if i<N else 1.0
    path_cost=lambda p: sum(etx(PKT_BITS,hyp(p[k],p[k+1]))+erx(PKT_BITS) for k in range(len(p)-1))
    def path_fitness(p,en):
        if len(p)<2: return -1e9
        rel=1.0; rew=[]
        for k in range(len(p)-1):
            a,b=p[k],p[k+1]; lq=linkq(a,b); rel*=max(lq,1e-3)
            rew.append(fuzzy_reward(node_attr(b,en),0.8,lq,hyp(a,b)/max(comm_r,1e-9)))
        e_ref=etx(PKT_BITS,hyp(p[0],SINK)+1e-9)*3+1e-12
        return 0.5*np.mean(rew)+0.3*rel+0.2*(1.0-min(1.0,path_cost(p)/e_ref))
    def valid(p,src):
        if not p or len(p)<2 or p[0]!=src or p[-1]!=SINK: return False
        if len(set(p))!=len(p): return False
        return all(p[k+1] in A[p[k]] for k in range(len(p)-1))
    def random_path(src,dst,maxlen):
        for _ in range(25):
            path=[src]; cur=src; s={src}
            for _ in range(maxlen):
                if dst in A[cur] and rng.random()<0.5: path.append(dst); return path
                nb=[v for v in A[cur] if v not in s and v!=dst]
                if dst in A[cur] and not nb: path.append(dst); return path
                if not nb: break
                w=np.array([1.0/(0.1+hyp(v,dst)) for v in nb]); w=w/w.sum()
                cur=nb[rng.choice(len(nb),p=w)]; s.add(cur); path.append(cur)
        return dijkstra(src,dst)
    def crossover(p1,p2):
        c=[x for x in p1[1:-1] if x in p2[1:-1]]
        if not c: return None
        x=c[rng.integers(len(c))]; ch=p1[:p1.index(x)]+p2[p2.index(x):]
        return ch if len(set(ch))==len(ch) else None
    def mutate(p):
        if len(p)<=2: return p
        i=1+rng.integers(len(p)-2); a,b=p[i-1],p[i+1]
        cand=[y for y in A[a] if (b in A[y]) and y not in p and y!=SINK]
        if not cand: return p
        q=list(p); q[i]=cand[rng.integers(len(cand))]; return q
    maxlen=2*int(math.sqrt(N+1))+5
    def ga_route(src,dst,en):
        pop=[]; sd=dijkstra(src,dst)
        if sd: pop.append(sd)
        while len(pop)<GA_POP:
            p=random_path(src,dst,maxlen)
            if p and valid(p,src): pop.append(p)
        for _ in range(GA_GEN):
            sc=sorted(pop,key=lambda p:-path_fitness(p,en))
            r=np.arange(len(sc),0,-1).astype(float); pr=r/r.sum(); nxt=sc[:max(2,GA_POP//5)]
            while len(nxt)<GA_POP:
                i,j=rng.choice(len(sc),2,p=pr)
                ch=crossover(sc[i],sc[j]) if rng.random()<GA_PCROSS else list(sc[i])
                if ch is None: ch=list(sc[i])
                if rng.random()<GA_PMUT: ch=mutate(ch)
                if valid(ch,src): nxt.append(ch)
            pop=nxt
        b=max(pop,key=lambda p:path_fitness(p,en))
        return b if valid(b,src) else (sd if sd else [src,dst])
    def per_node(emit,routes):
        E=np.full(N,E_SENSE,dtype=float)
        for s in range(N):
            p=routes[s]
            for k in range(len(p)-1):
                a,b=p[k],p[k+1]; d=hyp(a,b)
                if a<N: E[a]+=emit[s]*etx(PKT_BITS,d)
                if b<N: E[b]+=emit[s]*erx(PKT_BITS)
        return E
    def leach_energy(emit,draws=50,p=0.1):
        acc=np.zeros(N)
        for _ in range(draws):
            nCH=max(1,int(round(p*N))); CH=rng.choice(N,nCH,replace=False); E=np.full(N,E_SENSE,dtype=float)
            for i in range(N):
                if i in CH: continue
                c=CH[int(np.argmin([hyp(i,cc) for cc in CH]))]
                E[i]+=emit[i]*etx(PKT_BITS,hyp(i,c)); E[c]+=emit[i]*erx(PKT_BITS)
            for c in CH: E[c]+=emit[c]*etx(PKT_BITS,hyp(c,SINK))
            acc+=E
        return acc/draws
    E0=np.full(N+1,E_INIT); R={'direct':{},'shortest':{},'fga':{},'random':{}}
    for s in range(N):
        R['direct'][s]=[s,SINK]; sp=dijkstra(s,SINK); R['shortest'][s]=sp if sp else [s,SINK]
        R['fga'][s]=ga_route(s,SINK,E0); rp=random_path(s,SINK,maxlen)
        R['random'][s]=rp if (rp and valid(rp,s)) else [s,SINK]
    ones=np.ones(N); out={'N':N,'comm_r_m':round(comm_r,2)}
    for k in ['direct','shortest','fga','random']:
        E=per_node(ones,R[k]); out[k]={'net_life':N*E_INIT/E.sum(),'FND':E_INIT/E.max()}
    lp=max([0.05,0.10,0.15,0.20],key=lambda p:N*E_INIT/leach_energy(ones,50,p).sum())
    El=leach_energy(ones,50,lp); out['leach']={'net_life':N*E_INIT/El.sum(),'FND':E_INIT/El.max(),'best_p':lp}
    return out

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--locs",default="mote_locs.txt"); ap.add_argument("--meta",default="intel_time_meta.json")
    ap.add_argument("--motes",default=None,help="JSON list of mote ids (default: the motes in --meta)"); ap.add_argument("--seeds",default=",".join(map(str,range(20))))
    a=ap.parse_args()
    locs=pd.read_csv(a.locs,sep=r'\s+',names=['moteid','x','y']).set_index('moteid')
    motes=json.load(open(a.motes)) if a.motes else json.load(open(a.meta))["motes"]
    seeds=[int(x) for x in a.seeds.split(',')]
    pos=locs.loc[motes,['x','y']].to_numpy(dtype=float)
    runs=[{'seed':sd,**run(pos,sd)} for sd in seeds]
    summ={}
    for k in ['direct','shortest','fga','leach','random']:
        nl=np.array([r[k]['net_life'] for r in runs]); fn=np.array([r[k]['FND'] for r in runs])
        summ[k]=dict(net_life_mean=float(nl.mean()),net_life_sd=float(nl.std()),FND_mean=float(fn.mean()),FND_sd=float(fn.std()))
    summ['leach']['best_p']=sorted({r['leach']['best_p'] for r in runs})
    out=dict(motes=motes,seeds=seeds,summary=summ,runs=runs)
    fn=f"routing_{len(motes)}motes.json"; json.dump(out,open(fn,'w'),indent=1,default=float); print(json.dumps(summ,indent=1)); print('written',fn)
