"""
Version 1.4. Helper module for platform_table.py (Section 6.6, Table 6). Reuses the saved
per-seed test suppression (learned_<ds>.json) and the validation-selected cheap reference
(ms_numbers.json), at tau = 0.10 and the anchor e_r = 3.8e-4 J; nothing is retrained.

Provides: stream(ds) -> (reference, its test rho, its MACs, test rounds, per-seed rho of
each learned model); verdict(g) -> '+', '-' or 'ns' (Wilcoxon two-sided over the 20 seeds,
Bonferroni 0.05/22); work(m) = MACs + overheads in MAC-equivalents; PARAMS, REF_PARAMS.

ASSUMED overhead costs in MAC-equivalents (used for the processors whose E_op is taken
from PULP-NN, STM32L4 and STM32H7, and on GAP-8 only in the sensitivity run of gap8_lstm_energy.py;
the Cortex-M0 is simulated): LSTM elementwise product 1, sigmoid/tanh 2,
bias add or ReLU 0.25. Overhead counts are derived from the architectures.
The __main__ block, kept for the record, prints sensitivity runs (the 1 nJ check, and the PULP-NN
energies with 32-bit weights amortised over each test span; its GAP-8 row uses the PULP-NN figure,
which the paper no longer uses); the numbers of the paper come from platform_table.py.
"""
import json, numpy as np
from scipy.stats import wilcoxon

R = ""                       # run inside run/, next to ms_numbers.json and learned_*.json
M = json.load(open(R + "ms_numbers.json")); C = M["constants"]; D = M["datasets"]
E_R, E_S = C["E_R"], C["E_SENSE"]; THR = 0.05 / M["bonferroni_main_n"]
E_ELEC = 50e-9
DS = list(D.keys())

# ---- work per round (MAC-equivalents) ------------------------------------------
MACS = {"lightweight": 208, "cnn_lstm": 99488}
# overhead counts (DERIVED from the architectures in frozen_benchmark.build)
#  lightweight: Dense(16, ReLU) on 12 inputs, Dense(1): biases 16+1, ReLU 16
#  cnn_lstm: Conv1D(32,k=3,same,ReLU) on 12 steps: biases 384, ReLU 384;
#            LSTM(32) over 12 steps: gate biases 4*32*12 = 1536,
#            sigmoid 3*32*12 = 1152, tanh 2*32*12 = 768, elementwise mults 3*32*12 = 1152;
#            Dense(1): bias 1
OVH = {"lightweight": dict(ew=0, act=0, lin=16 + 1 + 16),
       "cnn_lstm":    dict(ew=1152, act=1152 + 768, lin=384 + 384 + 1536 + 1)}
PARAMS = {"lightweight": 12 * 16 + 16 + 16 + 1, "cnn_lstm": 3 * 32 + 32 + 4 * (32 * 32 + 32 * 32 + 32) + 32 + 1}
assert PARAMS == {"lightweight": 225, "cnn_lstm": 8481}
REF_PARAMS = {"persistence": 0, "expsm": 1, "linear_extrap": 1}
COST = dict(ew=1.0, act=2.0, lin=0.25)          # ASSUMED MAC-equivalents


def work(m, overhead=True):
    w = MACS[m]
    if overhead: w += sum(OVH[m][k] * COST[k] for k in OVH[m])
    return w

PLAT = {  # label: (E_op low, E_op high) in J per MAC-equivalent
    "GAP-8 [PULP-NN]":            (0.042e-9, 0.062e-9),
    "STM32L4 Cortex-M4 [PULP-NN]": (0.59e-9, 0.59e-9),
    "anchor (paper)":             (1.0e-9, 1.0e-9),
    "STM32H7 Cortex-M7 [PULP-NN]": (1.03e-9, 1.64e-9),
}


def stream(ds):
    b = D[ds]["by_tau"]["0.1"]; ref = b["selected"]; rr = b["test"][ref]["rho"]; rmac = b["selected_macs"]
    T = D[ds]["T"]; rounds = T - int(0.7 * T)
    ent = json.load(open(R + f"learned_{ds}.json"))["entries"]
    v = {m: np.array([e["test_rho"] for e in ent if e["predictor"] == m]) for m in MACS}
    return ref, rr, rmac, rounds, v


def verdict(g):
    if np.all(g == 0): return "0"
    p = wilcoxon(g).pvalue
    return ("+" if g.mean() > 0 else "-") if p < THR else "ns"


def table(eop_pick, overhead=True, transfer=True, bits=32):
    out = {}
    for ds in DS:
        ref, rr, rmac, rounds, v = stream(ds)
        rp = REF_PARAMS[ref.split("(")[0]] if ref.split("(")[0] in REF_PARAMS else 1
        for m in MACS:
            dW = work(m, overhead) - rmac
            dX = (PARAMS[m] - rp) * bits * E_ELEC / rounds if transfer else 0.0
            g = E_R * (v[m] - rr) - (dW * eop_pick + dX)
            out[(ds, m)] = (verdict(g), g.mean() * 1e6, dW * eop_pick * 1e6, dX * 1e6)
    return out


def summary(label, t):
    s = {}
    for m in MACS:
        vs = [t[(ds, m)][0] for ds in DS]
        s[m] = (vs.count("+"), vs.count("-"), vs.count("ns"))
    print(f"{label:46s} lightweight +{s['lightweight'][0]} -{s['lightweight'][1]} ns{s['lightweight'][2]} | "
          f"cnn_lstm +{s['cnn_lstm'][0]} -{s['cnn_lstm'][1]} ns{s['cnn_lstm'][2]}")
    return s


if __name__ == "__main__":
    print("work per round:", {m: (MACS[m], work(m)) for m in MACS}, "params:", PARAMS)
    print("transfer energy per node (fp32):", {m: PARAMS[m] * 32 * E_ELEC * 1e3 for m in MACS}, "mJ")
    for ds in DS:
        print(f"  {ds:17s} ref {stream(ds)[0]:16s} test rounds {stream(ds)[3]}")
    print()
    base = table(1e-9, overhead=False, transfer=False)
    summary("check: paper setting (1 nJ, no overheads)", base)
    for lab, (lo, hi) in PLAT.items():
        for e in sorted({lo, hi}):
            summary(f"{lab} @ {e*1e9:.3g} nJ + overheads + transfer", table(e))
    print()
    print("CNN-LSTM per stream (mean g in uJ, verdict):")
    for lab, (lo, hi) in PLAT.items():
        t = table(lo)
        print(f"  {lab:30s} @{lo*1e9:.3g}: " + " ".join(f"{ds[:6]}:{t[(ds,'cnn_lstm')][0]}{t[(ds,'cnn_lstm')][1]:+.0f}" for ds in DS))
