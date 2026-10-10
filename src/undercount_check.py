"""Version 1.4. The compute undercount quoted in the Introduction and Section 7 (paragraph 2) of the paper.
Inside run/, next to ms_numbers.json and learned_taxi.json (imports platform_energy.py and gap8_lstm_energy.py):
    python3 undercount_check.py            -> undercount_check.json

The CNN-LSTM charge of the paper, 99,488 MACs (frozen_benchmark.py), takes the 32 convolution channels as
the LSTM input. Counting the LSTM input as one channel, in the same way, gives
    Conv1D 12*32*3 + LSTM 4*32*(1+32)*12 + Dense 32 = 51,872 MACs, and 1 - 51,872/99,488 = 47.9% ("48%").
The 53,408 in two places of frozen_benchmark.py (a module docstring sentence and a comment) is an earlier
count. It is consistent with a bias term also counted as an LSTM input, 4*32*(1+32+1)*12 + 1,152 + 32, hence
its ~46%. Both undercounts are evaluated. An undercount is a counting error, not another model, so the
model keeps its 8,481 parameters.

Setting: taxi, tau = 0.10, anchor e_r = 3.8e-4 J, test span, the 20 floating-point seeds of learned_taxi.json.
Net saving per seed g = e_r*(rho_model - rho_ref) - (MACs_model - MACs_ref)*E_op - weights, against untuned
persistence (0 MACs) and against the validation-selected reference of ms_numbers.json (as platform_energy.stream).
At E_op = 1 nJ per MAC (the paper's main assumption) weights = 0, exactly as in ms_numbers.py. At the four
GAP-8 values of gap8_lstm_energy.py (Section 6.6) the weights are charged as there: received once as 16-bit,
amortised over 1e5 rounds. Verdict: Wilcoxon two-sided over the 20 seeds, Bonferroni 0.05/22. The statements
of the paper hold at 1 nJ per MAC and not at the GAP-8 values, which is why the paper states them at 1 nJ."""
import json, numpy as np
from scipy.stats import wilcoxon
import platform_energy as pe
import gap8_lstm_energy as g8

M = json.load(open("ms_numbers.json")); C = M["constants"]
b = M["datasets"]["taxi"]["by_tau"]["0.1"]; te = b["test"]
v = np.array([e["test_rho"] for e in json.load(open("learned_taxi.json"))["entries"] if e["predictor"] == "cnn_lstm"])
assert len(v) == 20

COUNTS = {"correct_32_channels": 12 * 32 * 3 + 4 * 32 * (32 + 32) * 12 + 32,
          "one_channel": 12 * 32 * 3 + 4 * 32 * (1 + 32) * 12 + 32,
          "earlier_comment": 4 * 32 * (1 + 32 + 1) * 12 + 12 * 32 * 3 + 32}
assert COUNTS["correct_32_channels"] == C["MACS"]["cnn_lstm"] == pe.MACS["cnn_lstm"] == 99488
assert COUNTS["one_channel"] == 51872 and COUNTS["earlier_comment"] == 53408
ref, rr, rmac, rounds, vv = pe.stream("taxi")
assert np.array_equal(vv["cnn_lstm"], v) and rr == te[b["selected"]]["rho"] and rmac == b["selected_macs"]
REFS = {"untuned_persistence": ("persistence", te["persistence"]["rho"], 0),
        "validation_selected_" + b["selected"]: (b["selected"], rr, rmac)}
EOPS = {"1.000": (1e-9, False), **{"%.3f" % (e * 1e9): (e, True) for e in g8.EOPS}}

def verdict(g):
    p = float(wilcoxon(g).pvalue)
    return dict(mean_uJ=round(float(g.mean()) * 1e6, 3), seeds_saving=int((g > 0).sum()), wilcoxon_p=p,
                verdict=("saving" if g.mean() > 0 else "loss") if p < pe.THR else "not significant")

out = dict(counts=COUNTS,
           undercount_fraction={k: round(1 - n / COUNTS["correct_32_channels"], 4) for k, n in COUNTS.items() if k != "correct_32_channels"},
           charge_rise_from_undercount={k: round(COUNTS["correct_32_channels"] / n - 1, 4) for k, n in COUNTS.items() if k != "correct_32_channels"},
           setting=dict(stream="taxi", tau=0.10, e_r=pe.E_R, seeds=len(v), bonferroni_threshold=pe.THR,
                        mean_rho_cnn_lstm=round(float(v.mean()), 4), weights_bits=g8.BITS, weights_rounds=g8.H),
           results={})
for rn, (rk, rho_ref, rm) in REFS.items():
    for cn, mac in COUNTS.items():
        for en, (eop, weights) in EOPS.items():
            w = (pe.PARAMS["cnn_lstm"] - pe.REF_PARAMS[rk]) * g8.BITS * pe.E_ELEC / g8.H if weights else 0.0
            out["results"][f"{rn}|{cn}|E_op={en}nJ"] = verdict(pe.E_R * (v - rho_ref) - (mac - rm) * eop - w)
json.dump(out, open("undercount_check.json", "w"), indent=1)
for k, r in out["results"].items(): print(f"{k:60s} {r['mean_uJ']:8.2f} uJ  {r['seeds_saving']:2d}/20  p={r['wilcoxon_p']:.1e}  {r['verdict']}")
