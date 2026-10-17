"""Moving-block bootstrap over test rounds (post hoc, descriptive).
For the three streams on which the lightweight model saves energy significantly across seeds (taxi, traffic,
exchange_cpc), the per-round net saving g_t = er*(sent_ref,t - sent_lw,t) - (M_lw - M_ref)*Eop is computed in exact
closed loop for each stored seed; test rounds are resampled in moving blocks of B rounds (the same blocks for every seed
and for the reference), and the statistic is the median over seeds of the mean net saving per node-round, as in the
paper's verdicts. Computed after the results (Section 5.2 and its supplement). Run from caps/boot after caps/setup.sh. Reported: point value, 95% percentile interval, share of replicates with a positive median."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, '../stage17/run')); WDIR = os.path.abspath(os.path.join(HERE, '../adaptA/weights'))
sys.path.insert(0, RUN)
import numpy as np
import frozen_benchmark as fb, fastfwd, nab_data as nd
ER, EOP, LB, TAU = fb.E_R, fb.E_OP, fb.LOOK_BACK, fb.TAU
MS = json.load(open(os.path.join(RUN, 'ms_numbers.json')))['datasets']

def ref_arm(ds):
    bt = MS[ds]['by_tau']['0.1']; sel = bt['selected']
    f = None if sel == 'persistence' else fb.make_expsm(bt['alpha']) if sel == 'expsm' else fb.make_ar(bt['K'])
    return sel, f, bt['selected_macs'], bt['test'][sel]['rho']

def learned_arm(A, wfile):
    T = A.shape[0]; tr = int(fb.SPLIT[0] * T)
    Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    w = np.load(wfile); P = ('lightweight', *[w['arr_%d' % i] for i in range(len(w.files))])
    return lambda Win: fastfwd.forward(P, (np.asarray(Win, float) - mx) / sx) * sy + my

def sent_series(X, f, unit):
    R = X.copy(); out = np.zeros(X.shape[0] - LB, bool)
    for t in range(LB, X.shape[0]):
        p = R[t - 1].copy() if f is None else np.asarray(f(R[t - LB:t].T), float).reshape(-1)
        sup = (np.abs(X[t] - p) / unit) <= TAU
        R[t] = np.where(sup, p, X[t]); out[t - LB] = not bool(sup[0])
    return out

def run(ds, Bs=(48, 96), nboot=10000):
    A, unit = nd.load(ds); A = np.asarray(A, float); A = A[:, None] if A.ndim == 1 else A
    T = A.shape[0]; va = int(fb.SPLIT[1] * T); X = A[va - LB:]
    sel, fR, mR, rho_ref = ref_arm(ds); sR = sent_series(X, fR, unit)
    assert abs((1 - sR.mean()) - rho_ref) < 1e-9, (ds, 1 - sR.mean(), rho_ref)
    mL = fb.MACS['lightweight']; stored = {r['seed']: r for r in json.load(open(os.path.join(RUN, f'learned_{ds}.json')))['entries'] if r['predictor'] == 'lightweight'}
    G = []; seeds = []
    for sd in range(20):
        wf = os.path.join(WDIR, f'{ds}_lightweight_s{sd}.npz')
        if not os.path.exists(wf): continue
        sL = sent_series(X, learned_arm(A, wf), unit)
        assert abs((1 - sL.mean()) - stored[sd]['test_rho']) <= 0.005, (ds, sd)
        G.append(ER * (sR.astype(float) - sL.astype(float)) - (mL - mR) * EOP); seeds.append(sd)
    G = np.array(G); n = G.shape[1]; point = float(np.median(G.mean(1)))
    rng = np.random.default_rng(20261017); res = {'ds': ds, 'ref': sel, 'n_rounds': int(n), 'seeds': seeds,
          'point_median_net_uJ': point * 1e6, 'per_seed_mean_uJ': (G.mean(1) * 1e6).round(3).tolist()}
    for B in Bs:
        nb = int(np.ceil(n / B)); starts = np.arange(0, n - B + 1)
        stats = np.empty(nboot)
        for i in range(nboot):
            idx = np.concatenate([np.arange(s, s + B) for s in rng.choice(starts, nb)])[:n]
            stats[i] = np.median(G[:, idx].mean(1))
        res[f'B{B}'] = {'ci95_uJ': (np.percentile(stats, [2.5, 97.5]) * 1e6).round(3).tolist(),
                        'share_positive': float((stats > 0).mean())}
    return res

if __name__ == '__main__':
    out = {ds: run(ds) for ds in ['taxi', 'traffic', 'exchange_cpc']}
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'block_bootstrap.json'), 'w'), indent=1)
    for ds, r in out.items():
        print(ds, r['ref'], 'n', r['n_rounds'], 'seeds', len(r['seeds']), 'median net %.2f uJ' % r['point_median_net_uJ'],
              {b: r[b] for b in ('B48', 'B96')})
