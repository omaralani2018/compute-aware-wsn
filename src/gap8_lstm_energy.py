"""Version 1.4. GAP-8 energy per MAC for an LSTM workload, derived from Cappellone et al., "On-board
satellite telemetry forecasting with RNN on RISC-V based multicore processor", IEEE DFT 2020
(doi 10.1109/DFT50435.2020.9250796; reference [27] of the paper), and the resulting verdicts at the anchor
(Section 6.6, Table 6). platform_table.py imports EOPS from this module for the GAP-8 column.
Inside run/, next to ms_numbers.json, learned_*.json and qcl_taxi_cnn_lstm.json:
    python3 gap8_lstm_energy.py            -> gap8_lstm_energy.json (after quant_closed_loop.py on taxi)

What is taken from the source (not measured here)
- Table I: parameter counts of its Seq2Seq LSTM forecaster (encoder and decoder LSTM layers, input
  dimension 1, dense output), used to recover the architecture: 2*4*(n(1+n)+n) + (n+1) parameters.
- Table IV: execution cycles per Seq2Seq prediction (27 encoder and 9 decoder steps) on the GAP-8 cluster,
  16-bit fixed point (Q3.12), weights in L2 for 60, 100 and 150 units.
- Text: cluster at 175 MHz; 75 mW quoted as the worst-case consumption (power is not measured there).
Derived here
- MACs per prediction = 36*4*(n + n^2) + 9n; MAC per cycle = MACs / cycles.
- Timing ambiguity: the "previsions/s" column of Table IV equals 9 outputs per prediction at 150 MHz in all
  ten rows (LSTM and GRU), not 175 MHz. Cycles, and so MAC per cycle, do not depend on the frequency;
  the energy does, through the assumed power. Two scenarios are therefore used: 175 MHz and 150 MHz,
  each with 65 mW (from the PULP-NN measurement on GAP-8, 0.27 mJ per inference at 241 per second) and
  75 mW, with no reduction of power at the lower frequency (conservative).
Assumptions (stated in the paper)
- Implementation transfer: our CNN-LSTM is assumed to run at the source's MAC per cycle. The source uses
  hard-sigmoid, its own tanh approximation and 32-bit biases; our 16-bit implementation (src/m0) uses
  interpolated tables. The 16-bit closed-loop check (qcl_*.json) covers our implementation, not theirs.
- The energy per MAC comes from whole-prediction cycles, so it already contains the source's activation,
  memory and control work: the main calculation charges MACs only (no MAC-equivalent overhead). Charging
  the overhead of platform_energy.py as well is reported as a sensitivity.
- Weights received once as 16-bit, amortised over 1e5 rounds (as for the Cortex-M0).
- Rate margin: our CNN-LSTM has 32 LSTM units, fewer than the 60-150 units whose rate is used, and smaller
  layers may run at a lower MAC per cycle. taxi_min_mac_per_cycle gives, for each power/clock scenario,
  the lowest MAC per cycle at which the taxi saving stays significant (worst case 75 mW at 150 MHz).
Verdicts: Wilcoxon over the 20 floating-point seeds (Bonferroni 0.05/22); the 15 fixed-point taxi seeds of
qcl_taxi_cnn_lstm.json are reported separately and are not pooled with them."""
import json, numpy as np
from scipy.stats import wilcoxon
import platform_energy as pe

params = lambda n: 2 * 4 * (n * (1 + n) + n) + (n + 1)
TABLE_I = {30: 7711, 60: 29821, 100: 81701, 150: 182551, 200: 323401}
assert all(params(n) == p for n, p in TABLE_I.items()), "architecture does not reproduce Table I"
macs = lambda n: 36 * 4 * (n + n * n) + 9 * n
TABLE_IV = {("LSTM", 60): (3.68e5, 3.67e3), ("LSTM", 100): (1.01e6, 1.34e3), ("LSTM", 150): (2.02e6, 6.68e2),
            ("LSTM", 200): (4.63e6, 2.92e2), ("LSTM", 250): (2.54e8, 0.53e1), ("GRU", 60): (2.74e5, 4.93e3),
            ("GRU", 100): (7.73e5, 1.75e3), ("GRU", 150): (1.59e6, 8.49e2), ("GRU", 200): (2.85e6, 4.73e2),
            ("GRU", 250): (2.46e8, 0.55e1)}
implied_mhz = {f"{t}{n}": round(pps * cyc / 9 / 1e6, 1) for (t, n), (cyc, pps) in TABLE_IV.items()}
assert all(abs(f - 150) < 1.5 for f in implied_mhz.values()), implied_mhz
mpc = {n: macs(n) / TABLE_IV[("LSTM", n)][0] for n in (60, 100, 150)}          # weights in L2
SCEN = {}
for f in (175e6, 150e6):
    lo, hi = 65e-3 / (f * max(mpc.values())), 75e-3 / (f * min(mpc.values()))
    SCEN[f"{int(f / 1e6)}MHz"] = (lo, hi)
EOPS = sorted({v for r in SCEN.values() for v in r})
H, BITS = 1e5, 16
REFMAC = {"persistence": 0, "expsm": 12, "linear_extrap": 2}

def stream_g(ds, m, eop, overhead, rho=None):
    ref, rr, rmac, rounds, v = pe.stream(ds); rk = ref.split("(")[0]
    work = pe.work(m) if overhead else pe.MACS[m]
    Em = work * eop + (pe.PARAMS[m] - pe.REF_PARAMS[rk]) * BITS * pe.E_ELEC / H
    Er = REFMAC[rk] * eop
    r = v[m] if rho is None else rho
    g = pe.E_R * (r - rr) - (Em - Er)
    L = (pe.E_S + Er + (1 - rr) * pe.E_R) / (pe.E_S + Em + (1 - np.mean(r)) * pe.E_R)
    return g, L

def last_significant(ds, m, overhead):
    a, b = 1e-12, 3e-9
    if pe.verdict(stream_g(ds, m, a, overhead)[0]) != "+": return None
    for _ in range(60):
        mid = (a + b) / 2
        if pe.verdict(stream_g(ds, m, mid, overhead)[0]) == "+": a = mid
        else: b = mid
    return a * 1e9

if __name__ == "__main__":
    out = dict(mac_per_cycle={str(n): mpc[n] for n in mpc}, implied_mhz_table_iv=implied_mhz,
               e_op_nJ={k: [v[0] * 1e9, v[1] * 1e9] for k, v in SCEN.items()}, verdicts={}, table6_gap8={})
    for overhead in (False, True):
        key = "main_no_overhead" if not overhead else "sensitivity_with_overhead"
        out["verdicts"][key] = {}
        for e in EOPS:
            V = {f"{ds}|{m}": pe.verdict(stream_g(ds, m, e, overhead)[0]) for ds in pe.DS for m in pe.MACS}
            out["verdicts"][key][f"{e * 1e9:.3f}"] = V
            print(f"{key:26s} {e * 1e9:.3f} nJ: CNN-LSTM saves {[k.split('|')[0] for k, x in V.items() if x == '+' and k.endswith('cnn_lstm')]}, "
                  f"lightweight saves {[k.split('|')[0] for k, x in V.items() if x == '+' and k.endswith('lightweight')]}, "
                  f"ns {[k for k, x in V.items() if x == 'ns']}")
    for ds in ("taxi", "exchange_cpc", "traffic", "intel_time"):
        for m in pe.MACS:
            Ls = [stream_g(ds, m, e, False)[1] for e in EOPS]
            out["table6_gap8"][f"{ds}|{m}"] = [min(Ls), max(Ls)]

    out["taxi_cnn_lstm_significant_up_to_nJ"] = {"no_overhead": last_significant("taxi", "cnn_lstm", False),
                                                 "with_overhead": last_significant("taxi", "cnn_lstm", True)}
    out["exchange_cpc_cnn_lstm_significant_up_to_nJ"] = {"no_overhead": last_significant("exchange_cpc", "cnn_lstm", False)}
    # Rate margin: our CNN-LSTM (32 LSTM units) is smaller than the 60-150-unit models whose rate is used.
    # Lowest MACs per cycle at which the taxi saving stays significant, per power/clock scenario:
    # E_op = P / (f * mpc) <= limit  <=>  mpc >= P / (f * limit).
    lim = out["taxi_cnn_lstm_significant_up_to_nJ"]
    out["taxi_min_mac_per_cycle"] = {
        f"{int(P * 1e3)}mW@{int(f / 1e6)}MHz": {k: P / (f * lim[k] * 1e-9) for k in ("no_overhead", "with_overhead")}
        for P in (65e-3, 75e-3) for f in (175e6, 150e6)}
    out["slowest_source_mac_per_cycle"] = min(mpc.values())
    q = json.load(open("qcl_taxi_cnn_lstm.json")); fx = np.array([r["fixed"] for r in q])
    ref, rr, rmac, rounds, v = pe.stream("taxi")
    assert all(abs(v["cnn_lstm"][r["seed"]] - r["float"]) < 1e-12 for r in q)
    out["taxi_fixed_point"] = {}
    for overhead in (False, True):
        for e in EOPS:
            g, _ = stream_g("taxi", "cnn_lstm", e, overhead, rho=fx)
            gf, _ = stream_g("taxi", "cnn_lstm", e, overhead)
            out["taxi_fixed_point"][f"{'with' if overhead else 'no'}_overhead|{e * 1e9:.3f}"] = dict(
                seeds=len(g), all_save=bool((g > 0).all()), min_uJ=float(g.min() * 1e6), max_uJ=float(g.max() * 1e6),
                wilcoxon_p=float(wilcoxon(g).pvalue), float_seeds=len(gf), float_mean_uJ=float(gf.mean() * 1e6),
                float_p=float(wilcoxon(gf).pvalue))
    print("MAC per cycle", {k: round(x, 3) for k, x in mpc.items()})
    print("E_op nJ", {k: [round(x * 1e9, 3) for x in v] for k, v in SCEN.items()})
    print("taxi significant up to", out["taxi_cnn_lstm_significant_up_to_nJ"], "| exchange_cpc up to", out["exchange_cpc_cnn_lstm_significant_up_to_nJ"])
    print("taxi min MAC per cycle", {k: {a: round(b, 3) for a, b in v.items()} for k, v in out["taxi_min_mac_per_cycle"].items()},
          "| slowest source rate", round(out["slowest_source_mac_per_cycle"], 3))
    for k, x in out["taxi_fixed_point"].items(): print("taxi fixed", k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in x.items()})
    print("Table 6 GAP-8", {k: [round(a, 3) for a in x] for k, x in out["table6_gap8"].items()})
    json.dump(out, open("gap8_lstm_energy.json", "w"), indent=1)
