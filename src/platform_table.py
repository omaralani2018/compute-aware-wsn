"""
Version 1.4. The numbers of Section 6.6 and the right block of Table 6. Inside run/, after
m0/measure.py has written the three m0_measure-*.json (copied to run/):
    python3 platform_table.py            -> platform_table.json

Platforms
  STM32L4, STM32H7: per-MAC energy DERIVED from the efficiencies (GMAC/s/W) REPORTED by PULP-NN [26],
      INT-8 network; E_inf = W * E_op with W = MACs + overheads (assumed MAC-equivalents:
      elementwise 1, sigmoid/tanh 2, bias/ReLU 0.25), weights sent as 8-bit.
  GAP-8: per-MAC energy DERIVED in gap8_lstm_energy.py from the cycle counts of a 16-bit LSTM
      forecaster [27] at an assumed 65-75 mW and 150-175 MHz (EOPS, 0.230-0.349 nJ per MAC); those
      cycles already include activation, memory and control work, so E_inf = MACs * E_op (no
      overheads); weights sent as 16-bit.
  Cortex-M0: per-inference energy SIMULATED (fixed-point C code, Nikov's Thumbulator
      teamplay-D4.5, Nikov's ten hardware-fitted models), three compiler builds
      (-O2, -O3, -O3 -funroll-loops) x ten configurations; weights sent as 16-bit.
  Model transfer: params * bits * E_elec (50 nJ/bit, Section 3.1) received once by each
      node, amortised over 1e5 rounds.
Verdict: per-seed g = e_r*drho - dE, Wilcoxon two-sided, Bonferroni 0.05/22, e_r = 3.8e-4 J.
L: Eq. (4') with the seed-mean suppression, sensing included.
"""
import json, numpy as np
import platform_energy as pe
import gap8_lstm_energy as g8

H = 1e5
SIM = {b: json.load(open(f)) for b, f in [("-O2", "m0_measure-O2.json"),
                                          ("-O3", "m0_measure-O3.json"),
                                          ("-O3 unroll", "m0_measure-O3-funroll-loops.json")]}
PULP = {"STM32L4": (0.59e-9, 0.59e-9), "STM32H7": (1.03e-9, 1.64e-9)}
REFNAME = lambda ref: ref.split("(")[0]


def cases(platform):
    """Yield (label, E(model_or_ref_name) -> J per inference, bits)."""
    if platform == "GAP-8":
        for e in g8.EOPS:
            def E(name, e=e):
                return (pe.MACS[name] if name in pe.MACS else {"persistence": 0, "expsm": 12, "linear_extrap": 2}[name]) * e
            yield f"GAP-8@{e*1e9:.3g}nJ", E, 16
    elif platform in PULP:
        for e in sorted(set(PULP[platform])):
            def E(name, e=e):
                if name in pe.MACS: return pe.work(name) * e
                return {"persistence": 0, "expsm": 12, "linear_extrap": 2}[name] * e   # taxi K = 2
            yield f"{platform}@{e*1e9:.3g}nJ", E, 8
    else:
        # the firmware's linear extrapolation uses K = 2, the K selected on taxi (2 MACs)
        assert all(pe.D[ds]["by_tau"]["0.1"]["selected_macs"] == 2 for ds in pe.DS
                   if pe.D[ds]["by_tau"]["0.1"]["selected"].startswith("linear_extrap"))
        for b, S in SIM.items():
            for cfg in S["cnn_lstm_q"]["energy_nJ"]:
                yield f"M0 {b} {cfg}", (lambda name, S=S, cfg=cfg: S[f"{name}_q"]["energy_nJ"][cfg] * 1e-9), 16


def evaluate(platform):
    V, L = {}, {}
    for lab, E, bits in cases(platform):
        for ds in pe.DS:
            ref, rr, rmac, rounds, v = pe.stream(ds); rk = REFNAME(ref)
            # sanity: reference MAC count used for the per-MAC platforms equals the paper's
            if platform in PULP or platform == "GAP-8": assert {"persistence": 0, "expsm": 12, "linear_extrap": 2}[rk] == rmac, (ds, rk, rmac)
            Er = E(rk)
            for m in pe.MACS:
                Em = E(m) + (pe.PARAMS[m] - pe.REF_PARAMS[rk]) * bits * pe.E_ELEC / H
                g = pe.E_R * (v[m] - rr) - (Em - Er)
                V.setdefault((m, ds), []).append(pe.verdict(g))
                Lv = (pe.E_S + Er + (1 - rr) * pe.E_R) / (pe.E_S + Em + (1 - v[m].mean()) * pe.E_R)
                L.setdefault((m, ds), []).append(Lv)
    return V, L


if __name__ == "__main__":
    out = {}
    for P in ["GAP-8", "STM32L4", "STM32H7", "Cortex-M0"]:
        V, L = evaluate(P); out[P] = {}
        for m in pe.MACS:
            plus = [ds for ds in pe.DS if set(V[(m, ds)]) == {"+"}]
            minus = [ds for ds in pe.DS if set(V[(m, ds)]) == {"-"}]
            other = {ds: sorted(set(V[(m, ds)])) for ds in pe.DS if ds not in plus and ds not in minus}
            out[P][m] = dict(saves=plus, costs=len(minus), other=other)
            print(f"{P:9s} {m:11s} saves {plus} costs {len(minus)} other {other}")
        for ds in ["taxi", "exchange_cpc", "traffic", "intel_time"]:
            for m in pe.MACS:
                lo, hi = min(L[(m, ds)]), max(L[(m, ds)])
                out[P][f"L_{ds}_{m}"] = [lo, hi]
                print(f"   L {ds:12s} {m:11s} {lo:.3f}" + ("" if round(lo, 3) == round(hi, 3) else f"-{hi:.3f}"))
    # M0 headline numbers
    e = [S[f"cnn_lstm_q"]["energy_min_nJ"] for S in SIM.values()] + [S["cnn_lstm_q"]["energy_max_nJ"] for S in SIM.values()]
    l = [S["lightweight_q"]["energy_min_nJ"] for S in SIM.values()] + [S["lightweight_q"]["energy_max_nJ"] for S in SIM.values()]
    out["M0_cnn_uJ"] = [min(e) / 1e3, max(e) / 1e3]; out["M0_light_uJ"] = [min(l) / 1e3, max(l) / 1e3]
    out["M0_cnn_reports"] = [min(e) * 1e-9 / pe.E_R, max(e) * 1e-9 / pe.E_R]
    out["M0_cnn_nJ_per_MAC"] = [min(e) / 99488, max(e) / 99488]
    out["transfer_mJ"] = {m: {b: pe.PARAMS[m] * b * pe.E_ELEC * 1e3 for b in (8, 16, 32)} for m in pe.MACS}
    print(json.dumps({k: out[k] for k in ["M0_cnn_uJ", "M0_light_uJ", "M0_cnn_reports", "M0_cnn_nJ_per_MAC", "transfer_mJ"]}, indent=1))
    json.dump(out, open("platform_table.json", "w"), indent=1)
