"""Table 4: lightweight minus validation-selected reference at each tau; Bonferroni over 44."""
import json, numpy as np
import os; WORKDIR = os.path.abspath(os.environ.get("WORKDIR", "."))   # the directory the runs were made in (see README)
from scipy.stats import wilcoxon
DS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
      "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
TAUS = ["0.05", "0.1", "0.2", "0.5"]
cheap = json.load(open(f"{WORKDIR}/audit2/cheap.json"))
BONF = 0.05 / 44
print(f"Bonferroni/44 threshold {BONF:.5f}")
better = worse = inconcl = 0
never = []       # streams that never favour the learned model (no positive mean at any tau)
never_sig = []   # streams with no significant positive at any tau
always_sig_pos = []
letters = {"persistence": "P", "expsm": "E", "linear_extrap": "L"}
arm_changes = 0
tau01_match = True
for ds in DS:
    sw = json.load(open(f"{WORKDIR}/tau_{ds}.json"))
    assert sorted(e["seed"] for e in sw) == list(range(20)), ds
    sw = sorted(sw, key=lambda e: e["seed"])
    learned = json.load(open(f"{WORKDIR}/learned_{ds}.json"))
    lw = sorted([e for e in learned["entries"] if e["predictor"] == "lightweight"], key=lambda e: e["seed"])
    # consistency of the tau=0.1 sweep entries with learned_*.json lightweight
    for e, l in zip(sw, lw):
        if abs(e["0.1"]["test"] - l["test_rho"]) > 1e-12 or e["0.1"]["epoch"] != l["selected_epoch"]:
            tau01_match = False
            print(f"  NOTE {ds} seed {e['seed']}: tau-sweep 0.1 test {e['0.1']['test']:.6f} ep {e['0.1']['epoch']} vs learned {l['test_rho']:.6f} ep {l['selected_epoch']}")
    line = f"{ds:18s}"
    arms = []
    pos_any = False; sigpos_all = True; sigpos_any = False
    for tau in TAUS:
        ref = cheap[ds]["taus"][{"0.1": "0.1", "0.05": "0.05", "0.2": "0.2", "0.5": "0.5"}[tau]]["reference"]
        rho = np.array([e[tau]["test"] for e in sw])
        dd = rho - ref["test"]
        p = wilcoxon(dd, alternative="two-sided").pvalue if np.any(dd != 0) else 1.0
        sig = p < BONF
        if sig and dd.mean() > 0: better += 1
        elif sig and dd.mean() < 0: worse += 1
        else: inconcl += 1
        if dd.mean() > 0: pos_any = True
        if sig and dd.mean() > 0: sigpos_any = True
        else: sigpos_all = False
        arms.append(ref["arm"])
        line += f"  {dd.mean():+.3f}{'*' if sig else ' '} ({letters[ref['arm']]}) p={p:.1e}"
        # verified flags and max_err bound in the sweep
        for e in sw:
            assert e[tau]["verified"] is True, (ds, tau, e["seed"])
            assert e[tau]["max_err"] <= float(tau) * cheap[ds]["unit"] + 1e-9, (ds, tau, e["seed"], e[tau]["max_err"])
    if not pos_any: never.append(ds)
    if not sigpos_any: never_sig.append(ds)
    if sigpos_all: always_sig_pos.append(ds)
    if len(set(arms)) > 1: arm_changes += 1
    print(line)
print(f"\nbetter={better} worse={worse} inconclusive={inconcl} (of 44)")
print("streams with no positive mean gain at any tau:", never, len(never))
print("streams with no significant positive gain at any tau:", never_sig, len(never_sig))
print("streams significantly positive at every tau:", always_sig_pos)
print("streams where the reference arm changes with tau:", arm_changes, "of 11")
print("tau=0.1 sweep entries identical to learned_*.json lightweight:", tau01_match)
