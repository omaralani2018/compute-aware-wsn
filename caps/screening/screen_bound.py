"""Training-free screening bound (analysis on shipped results only; no new runs).
Necessary condition for a candidate to pay against the reference: Einf - Einf_ref < (1 - rho_ref) * er,
because rho <= 1 so delta_rho <= 1 - rho_ref. rho_ref is taken from the VALIDATION span (known before any
learned model is trained). Usage: python3 -I screen_bound.py <ms_numbers.json>"""
import json, sys
d = json.load(open(sys.argv[1]))['datasets']
ER, EOP = 3.8e-4, 1e-9
MACS = {'lightweight': 208, 'cnn_lstm': 99488}
rows = []
for s, v in d.items():
    b = v['by_tau']['0.1']; sel = b['selected']; m_ref = b['selected_macs']
    rv = b['val'][sel]; rt = b['test'][sel]['rho']
    out = {'stream': s, 'ref': sel, 'ref_macs': m_ref, 'rho_ref_val': rv, 'rho_ref_test': rt,
           'budget_macs_val': ((1 - rv) * ER) / EOP + m_ref}
    for m, M in MACS.items():
        L = v['learned'][m]
        need = (M - m_ref) * EOP / ER                     # suppression gain needed to break even
        excluded = (M - m_ref) * EOP >= (1 - rv) * ER      # bound says: cannot pay, do not train
        er_lb = (M - m_ref) * EOP / (1 - rv)               # lower bound on the crossover radio cost
        out[m] = {'needed_gain': need, 'excluded_by_bound': excluded, 'er_lower_bound': er_lb,
                  'actual_net_saving': L['net_saving'], 'actual_sig_net': L['sig_net'],
                  'actual_crossover': L.get('crossover'), 'actual_delta': L['delta']}
    rows.append(out)
json.dump(rows, open('screen_bound.json', 'w'), indent=1)
for r in rows:
    c = r['cnn_lstm']; l = r['lightweight']
    print(f"{r['stream']:17s} ref={r['ref']:13s} rho_val={r['rho_ref_val']:.3f} rho_test={r['rho_ref_test']:.3f} budget={r['budget_macs_val']:8.0f} MAC | "
          f"CNN excl={c['excluded_by_bound']!s:5s} net={c['actual_net_saving']*1e6:7.1f}uJ er*={c['actual_crossover'] if c['actual_crossover'] is not None else 'never'} lb={c['er_lower_bound']:.2e} | "
          f"LW excl={l['excluded_by_bound']!s:5s} net={l['actual_net_saving']*1e6:6.1f}uJ")
