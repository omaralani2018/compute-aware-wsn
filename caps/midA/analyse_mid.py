"""PREREG14 A analysis, written before any mid-size result was inspected."""
import os, sys, json, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
RUN = os.path.join(SC, "stage17/run")
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]
DS = list(MS)
ER, EOP, ALPHA = 3.8e-4, 1e-9, 0.05
MACS = {"lightweight": 208, "mlp_mid": 2848, "cnnlstm_small": 6440, "cnn_lstm": 99488}
NJ = {  # measured incremental nJ per MAC of Section 6.14, by architecture family (secondary, labelled estimate)
    "M3 float": {"mlp": 155.36, "cnn": 177.55}, "M3 integer": {"mlp": 34.68, "cnn": 32.02}, "nRF52840": {"mlp": 6.39, "cnn": 7.66}}
FAM = {"lightweight": "mlp", "mlp_mid": "mlp", "cnnlstm_small": "cnn", "cnn_lstm": "cnn"}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    for i, k in enumerate(o):
        if p[k] <= ALPHA / (m - i): rej[k] = True
        else: break
    return rej
def pw(x, alt):
    x = np.asarray(x, float); return 1.0 if np.all(x == 0) else float(wilcoxon(x, alternative=alt).pvalue)

rho = {}
for ds in DS:
    L = json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"]
    rho[ds] = {k: {e["seed"]: e for e in L if e["predictor"] == k} for k in ("lightweight", "cnn_lstm")}
    for k in ("mlp_mid", "cnnlstm_small"):
        E = json.load(open(os.path.join(HERE, f"learned_{ds}_{k}.json")))["entries"]
        assert len(E) == 20 and all(e["receiver_verified"] for e in E), (ds, k, len(E))
        rho[ds][k] = {e["seed"]: e for e in E}
def t(ds, k): return [e["test_rho"] for e in rho[ds][k].values()]

def mw(a, b): return float(mannwhitneyu(t(ds, a), t(ds, b), alternative="two-sided").pvalue)
out = {"per_stream": {}}
pairs = {"arch": ("mlp_mid", "cnnlstm_small"), "size_mlp": ("mlp_mid", "lightweight"), "size_cnn": ("cnnlstm_small", "cnn_lstm")}
P = {n: [] for n in pairs}; D = {n: [] for n in pairs}
for ds in DS:
    for n, (a, b) in pairs.items():
        P[n].append(mw(a, b)); D[n].append(st.median(t(ds, a)) - st.median(t(ds, b)))
sig = {n: holm(P[n]) for n in pairs}
cnt = {n: dict(sig=int(sig[n].sum()), first_higher=int(sum(sig[n][i] and D[n][i] > 0 for i in range(len(DS)))),
               second_higher=int(sum(sig[n][i] and D[n][i] < 0 for i in range(len(DS))))) for n in pairs}
# energy verdicts at the anchor, as Section 6.3, and the screen
ver = {}; scr_false = []
for k in MACS:
    ps, pc, nets = [], [], []
    for ds in DS:
        bt = MS[ds]["by_tau"]["0.1"]; tR = bt["test"][bt["selected"]]["rho"]; vR = bt["val"][bt["selected"]]; mR = bt["selected_macs"]
        net = [(r - tR) * ER - (MACS[k] - mR) * EOP for r in t(ds, k)]
        ps.append(pw(net, "greater")); pc.append(pw(net, "less")); nets.append(st.median(net))
        screened = (MACS[k] - mR) * EOP >= (1 - vR) * ER
        if screened and st.median(net) > 0: scr_false.append((ds, k))
    s, c = holm(ps), holm(pc)
    ver[k] = dict(save=int(s.sum()), cost=int(c.sum()), save_streams=[DS[i] for i in range(len(DS)) if s[i]],
                  median_net_uJ={DS[i]: nets[i] * 1e6 for i in range(len(DS))})
# secondary: platform estimate (positive median net streams)
plat = {}
for pl, nj in NJ.items():
    plat[pl] = {}
    for k in MACS:
        pos = 0
        for ds in DS:
            bt = MS[ds]["by_tau"]["0.1"]; tR = bt["test"][bt["selected"]]["rho"]; mR = bt["selected_macs"]
            e = MACS[k] * nj[FAM[k]] * 1e-9; eR = mR * nj["mlp"] * 1e-9
            pos += st.median([(r - tR) * ER - (e - eR) for r in t(ds, k)]) > 0
        plat[pl][k] = int(pos)
for i, ds in enumerate(DS):
    print("%-17s median test rho: lw %.3f mlp_mid %.3f cnn_small %.3f cnn %.3f | arch p=%.3g size_mlp p=%.3g size_cnn p=%.3g" % (
        ds, *[st.median(t(ds, k)) for k in ("lightweight", "mlp_mid", "cnnlstm_small", "cnn_lstm")], P["arch"][i], P["size_mlp"][i], P["size_cnn"][i]))
print("significant (Holm): ", cnt)
print("energy verdicts at the anchor (save / cost):", {k: (v["save"], v["cost"], v["save_streams"]) for k, v in ver.items()})
print("screen false exclusions:", scr_false or "none")
print("secondary estimate, streams with positive median net:", plat)
a = cnt["arch"]["sig"]; sm = cnt["size_mlp"]["sig"]; sc = cnt["size_cnn"]["sig"]
if a <= 2 and max(sm, sc) >= 6: concl = "size, not architecture"
elif max(cnt["arch"]["first_higher"], cnt["arch"]["second_higher"]) >= 6: concl = "architecture"
else: concl = "mixed"
stand = max(ver["mlp_mid"]["save"], ver["cnnlstm_small"]["save"]) <= 3 and not scr_false
print("CONCLUSION:", concl, "| selection conclusions stand:", stand)
json.dump(dict(counts=cnt, p=P, d=D, verdicts=ver, screen_false=scr_false, platform_estimate=plat, conclusion=concl, stand=stand),
          open(os.path.join(HERE, "mid_summary.json"), "w"), indent=1, default=float)
