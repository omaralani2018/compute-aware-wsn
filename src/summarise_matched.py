"""
Summaries of the matched comparison (version 1.2): the numbers of Table 10 and
of the Section 6.8 paragraph.

Inside run/, after matched_baselines.py has written the eleven matched_*.json
and lstmdps_protocol.py has written lstmdps_intel_runs.jsonl:
  python3 summarise_matched.py
reads ms_numbers.json (the reference arm of each stream) and intel_time.npy,
and writes
  matched_summary.json   LMS and AM-DR against the reference, per stream and rule,
                         the derived counts quoted in Section 6.8, and the
                         published settings at 0.50 C on the Intel test span
  lstmdps_summary.json   LSTM-DPS against persistence on the Intel test span
Energy: g = e_r * (rho - rho_ref) - (MACs - MACs_ref) * E_op, per node-round,
at the anchor e_r = 3.8e-4 J and E_op = 1 nJ per MAC. SDs are population SDs
(ddof = 0), the convention of the paper.
"""
import json, numpy as np
from scipy.stats import wilcoxon
import matched_baselines as mb

E_R, E_OP = mb.E_R, mb.E_OP
DS = ['intel_time', 'taxi', 'traffic', 'cpu', 'machine_temp', 'grok_asg_anomaly', 'rds_cpu_util',
      'iio_networkin', 'twitter_aapl', 'exchange_cpc', 'occupancy']


def lms_amdr():
    m = json.load(open('ms_numbers.json'))['datasets']
    out = {}
    print(f"{'dataset':17s} {'ref':17s} {'refρ':>6s} | {'LMS-a':>6s} {'LMS-o':>6s} | {'AMDR-a':>6s} {'AMDR-o':>6s}")
    for ds in DS:
        r = json.load(open(f'matched_{ds}.json')); b = m[ds]['by_tau']['0.1']
        sel = b['selected']; refrho = b['test'][sel]['rho']; refmac = b['selected_macs']
        d = {(x['scheme'], x['rule']): x for x in r['runs']}
        o = {'ref': sel, 'ref_macs': refmac, 'ref_rho': refrho, 'pers_rho': b['test']['persistence']['rho']}
        for k, x in d.items():
            assert x['test']['receiver_identical'] and x['test']['bound_holds']
            allzero = all(v == 0 for _, v in x['val_grid'])
            dr = x['test']['rho'] - refrho; dEinf = (x['macs'] - refmac) * E_OP
            o[f"{k[0]}_{k[1]}"] = {'rho': x['test']['rho'], 'drho': dr, 'macs': x['macs'],
                                   'g_uJ': (E_R * dr - dEinf) * 1e6, 'sel': x['selected'],
                                   'val_all_zero': allzero,
                                   'max_err_over_tau_unit': x['test']['max_err'] / (mb.TAU * r['unit'])}
        out[ds] = o
        f = lambda k: d[k]['test']['rho']
        print(f"{ds:17s} {sel + '(' + str(refmac) + ')':17s} {refrho:6.4f} | {f(('LMS', 'authors')):6.4f} "
              f"{f(('LMS', 'ours')):6.4f} | {f(('AMDR', 'authors')):6.4f} {f(('AMDR', 'ours')):6.4f}")
    return out


def derived(out):
    """Counts quoted in Section 6.8 ("authors" = released rule, "ours" = single-error rule)."""
    rise, closed, beats, max_err = [], 0, {'LMS': [], 'AMDR': []}, []
    for ds, o in out.items():
        for sc in ('LMS', 'AMDR'):
            a, s, ref = o[sc + '_authors']['rho'], o[sc + '_ours']['rho'], o['ref_rho']
            rise.append(s - a)
            if s >= ref or (s - a) > 0.5 * (ref - a):        # more than half of the gap closed
                closed += 1
            if s > ref:
                beats[sc].append({'stream': ds, 'drho': o[sc + '_ours']['drho'], 'g_uJ': o[sc + '_ours']['g_uJ'],
                                  'macs': o[sc + '_ours']['macs']})
            for rule in ('authors', 'ours'):
                max_err.append(o[f'{sc}_{rule}']['max_err_over_tau_unit'])
    released_below_ref = all(o[sc + '_authors']['rho'] < o['ref_rho'] for o in out.values() for sc in ('LMS', 'AMDR'))
    return {'released_rule_below_reference_on_every_stream': released_below_ref,
            'single_error_rule_rise_min': min(rise), 'single_error_rule_rise_max': max(rise),
            'single_error_rule_rises_everywhere': min(rise) > 0,
            'pairs_closing_more_than_half_the_gap': closed, 'pairs': len(rise),
            'single_error_beats_reference': {k: {'streams': len(v), 'detail': v} for k, v in beats.items()},
            'all_gains_pay_at_anchor': all(b['g_uJ'] > 0 for v in beats.values() for b in v),
            'max_drho_of_gains': max(b['drho'] for v in beats.values() for b in v),
            'macs_range_LMS_AMDR': [min(o[k]['macs'] for o in out.values() for k in o if k.startswith(('LMS', 'AMDR'))),
                                    max(o[k]['macs'] for o in out.values() for k in o if k.startswith(('LMS', 'AMDR')))],
            'largest_sink_error_over_tau': max(max_err)}


def published_at(A, tau):
    """Published settings of [13] and [25] on the Intel test span at tolerance tau (C)."""
    T = A.shape[0]; va = int(mb.SPLIT[1] * T); X = A[va - mb.LOOK_BACK:]; n = A.shape[1]
    arms = {'LMS_N5_mu1e-5': (lambda w: mb.LMS(n, 5, 1e-5, w), 5),
            'AMDR_wf5_ws10_a1e-7': (lambda w: mb.AMDR(n, 5, 10, 1e-7, w), 5),
            'persistence': (lambda w: mb.Persistence(n, w), 1)}
    out = {'tau': tau}
    for name, (mk, W) in arms.items():
        for rule in ('authors', 'ours'):
            o = mb.run_scheme(X, mk, rule, W, tau=tau, return_packets=True)
            rec = mb.receiver(X, mk, o['_packets'], tau, 1.0)
            out[f'{name}_{rule}'] = {'rho': o['rho'], 'receiver_identical': bool(np.array_equal(rec, o['_recon'])),
                                     'bound_holds': bool(np.abs(rec[mb.LOOK_BACK:] - X[mb.LOOK_BACK:]).max() <= tau + 1e-12)}
    return out


def lstmdps(A):
    T = len(A); va = int(mb.SPLIT[1] * T)
    o = mb.run_scheme(A[va - 12:], lambda w: mb.Persistence(A.shape[1], w), 'ours', 1, return_packets=True)
    pers = 1 - o['_packets'][12:].mean(0)
    rows = [json.loads(l) for l in open('lstmdps_intel_runs.jsonl')]
    assert all(r['receiver_identical'] and r['bound_holds'] for r in rows)
    seeds = sorted({r['seed'] for r in rows})
    R = np.full((len(seeds), A.shape[1]), np.nan); M = R.copy()
    for r in rows:
        R[seeds.index(r['seed']), r['mote_col']] = r['rho']; M[seeds.index(r['seed']), r['mote_col']] = r['macs']
    assert np.isfinite(R).all(), 'incomplete runs'
    net = R.mean(1); macs = M.mean(1)
    g = E_R * (net - pers.mean()) - macs * E_OP          # J per node-round against persistence (0 MACs)
    d = R.mean(0) - pers
    out = {'n_runs': len(rows), 'seeds': seeds, 'net_rho': net.tolist(), 'pers_rho': float(pers.mean()),
           'mean_macs': macs.tolist(), 'g_uJ': (g * 1e6).tolist(),
           'per_mote_delta_mean': float(d.mean()), 'motes_better': int((d > 0).sum()), 'motes': int(d.size),
           'per_mote_wilcoxon_p': float(wilcoxon(d).pvalue),
           'net_rho_mean': float(net.mean()), 'net_rho_sd_pop': float(net.std()),
           'macs_min': int(np.nanmin(M)), 'macs_max': int(np.nanmax(M))}
    print('LSTM-DPS network rho %.4f +/- %.4f (pop SD) vs persistence %.4f; g %s uJ; MACs %d-%d'
          % (net.mean(), net.std(), pers.mean(), np.round(g * 1e6, 2), out['macs_min'], out['macs_max']))
    return out


if __name__ == '__main__':
    A = np.load('intel_time.npy')
    s = lms_amdr()
    s['_derived'] = derived(s)
    s['_published_settings_intel_tau0.5'] = published_at(A, 0.5)
    print(json.dumps(s['_derived'], indent=1))
    print({k: round(v['rho'], 4) for k, v in s['_published_settings_intel_tau0.5'].items() if k != 'tau'})
    json.dump(s, open('matched_summary.json', 'w'), indent=1)
    json.dump(lstmdps(A), open('lstmdps_summary.json', 'w'), indent=1)
