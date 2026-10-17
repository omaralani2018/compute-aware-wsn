"""Version 1.6 record (written 14 Oct 2026, after the results; reads only shipped outputs).
PREREG14 A asks for the energy verdict of each model "as in Section 6.3 (one-sided Wilcoxon over seeds in each
direction, Holm across streams)", and analyse_mid.py applies that rule. Section 6.3 itself uses a two-sided Wilcoxon
test judged against the Bonferroni threshold 0.05/22, the direction given by the sign of the mean net saving. This
script counts, for the four models, the streams with a significant saving and a significant cost under both rules."""
import os, json, statistics as st
import numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
RUN = os.path.join(SC, "stage17/run")
MS = json.load(open(os.path.join(RUN, "ms_numbers.json")))["datasets"]; DS = list(MS)
ER, EOP, ALPHA = 3.8e-4, 1e-9, 0.05
MACS = {"lightweight": 208, "mlp_mid": 2848, "cnnlstm_small": 6440, "cnn_lstm": 99488}

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); rej = np.zeros(m, bool)
    for i, k in enumerate(o):
        if p[k] <= ALPHA / (m - i): rej[k] = True
        else: break
    return rej
def pw(x, alt):
    x = np.asarray(x, float); return 1.0 if np.all(x == 0) else float(wilcoxon(x, alternative=alt).pvalue)
def rhos(ds, k):
    if k in ("lightweight", "cnn_lstm"):
        L = json.load(open(os.path.join(RUN, f"learned_{ds}.json")))["entries"]
        return [e["test_rho"] for e in L if e["predictor"] == k]
    return [e["test_rho"] for e in json.load(open(os.path.join(HERE, f"learned_{ds}_{k}.json")))["entries"]]

out = {}
for k in MACS:
    ps, pc, p2, mean = [], [], [], []
    for ds in DS:
        bt = MS[ds]["by_tau"]["0.1"]; tR = bt["test"][bt["selected"]]["rho"]; mR = bt["selected_macs"]
        net = [(r - tR) * ER - (MACS[k] - mR) * EOP for r in rhos(ds, k)]
        assert len(net) == 20, (ds, k)
        ps.append(pw(net, "greater")); pc.append(pw(net, "less")); p2.append(pw(net, "two-sided")); mean.append(st.mean(net))
    s, c = holm(ps), holm(pc)
    b = [p <= ALPHA / 22 for p in p2]
    out[k] = {"prereg14_holm_one_sided": {"save": [DS[i] for i in range(len(DS)) if s[i]], "cost": int(c.sum())},
              "section63_bonferroni22_two_sided": {"save": [DS[i] for i in range(len(DS)) if b[i] and mean[i] > 0],
                                                   "cost": int(sum(b[i] and mean[i] < 0 for i in range(len(DS))))}}
    h, f = out[k]["prereg14_holm_one_sided"], out[k]["section63_bonferroni22_two_sided"]
    print("%-14s Holm one-sided: save %d %s cost %d | Bonferroni/22 two-sided: save %d %s cost %d" % (
        k, len(h["save"]), h["save"], h["cost"], len(f["save"]), f["save"], f["cost"]))
json.dump(out, open(os.path.join(HERE, "verdict_rule_check.json"), "w"), indent=1)
