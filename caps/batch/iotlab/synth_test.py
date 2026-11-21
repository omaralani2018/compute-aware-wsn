"""Synthetic check of analyse_batch.py: builds a serial log and a power trace from a known energy model with the
firmware's phase sequence, a clock offset and noise, and checks that the analysis recovers the model."""
import json, numpy as np, sys
rng=np.random.default_rng(1); S=json.load(open('sched.json'))
P_IDLE=0.100; OFF=7.3
efr=lambda b: 2.6e-4+3.7e-6*(b-4)          # true frame energy above idle (J)
tfr=lambda b: 2.0e-3+3.2e-5*(b-4)          # time per frame (s)
WAKE=4e-6                                   # energy above idle per reading wake-up (J)
seg=[];ser=[];t=100.0
def emit(m): ser.append('%.6f;m3-21;%s'%(t,m))
def phase(kind,rnd,dur,Pavg,extra=''):
    global t
    emit('B %s begin %d t=0'%(('RUN '+kind) if kind!='IDLE' else 'IDLE',rnd)); seg.append((t,t+dur,Pavg)); t+=dur
    emit('B %s end %d t=0 %s'%(('RUN '+kind) if kind!='IDLE' else 'IDLE',rnd,extra))
emit('This is RIOT!'); emit('HDR board=iotlab-m3 role=TX ch=22'); ser.append('%.6f;m3-22;HDR board=iotlab-m3 role=RX'%t)
phase('IDLE',-1,2,P_IDLE); phase('SYNC',-1,12,P_IDLE+0.05); phase('IDLE',-1,2,P_IDLE)
cells=list(S['cells'])
RX={}
for r in range(5):
    phase('IDLE',r,2,P_IDLE)
    for i,b in enumerate([4,20,40,60,80,100]):
        n=int(4/tfr(b)); phase('SW%d'%b,r,4,P_IDLE+efr(b)*n/4,'calls=%d wall_ms=4000 send_err=0'%n); RX[(i,r)]=n; phase('IDLE',r,2,P_IDLE)
    for j,nm in enumerate(cells+['NULL']):
        nr=S['cells'][nm]['readings'] if nm!='NULL' else 1500; fs=S['frame_sizes'].get(nm,[])
        dur=nr*0.02; E=sum(efr(b) for b in fs)+WAKE*nr
        phase('R_'+nm,r,dur,P_IDLE+E/dur,'calls=%d reads=%d wall_ms=%d send_err=0 late=0'%(len(fs),nr,int(dur*1000))); RX[(6+j,r)]=len(fs)
        phase('IDLE',r,2,P_IDLE)
for (v,r),n in RX.items():
    if n: ser.append('%.6f;m3-22;RXV %d %d %d'%(t,v,r,n))
open('synth_serial.txt','w').write('\n'.join(ser)+'\n')
ts=np.arange(90,t+10,0.002); Pw=np.full(len(ts),P_IDLE)
for a,b,p in seg: Pw[(ts>=a+OFF)&(ts<b+OFF)]=p
Pw+=rng.normal(0,0.004,len(ts))
with open('synth.oml','w') as f:
    f.write('schema: x\n\n')
    for x,p in zip(ts,Pw): f.write('0\t1\t0\t%d\t%d\t%.6f\t3.3\t0.03\n'%(int(x),int(round((x-int(x))*1e6)),p))
json.dump(dict(variants=dict(S_BC=dict(E_inc_J=efr(4)),L_BC=dict(E_inc_J=efr(100)))),open('synth_r15.json','w'))
print('true frame E 4B %.3e 100B %.3e; offset %.1f'%(efr(4),efr(100),OFF))
