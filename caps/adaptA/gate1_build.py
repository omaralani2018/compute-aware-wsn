"""Version 1.6: rebuilds gate1_summary.json (Gate 1 of PREREG.txt) from oracle_<ds>_lightweight.json.
Columns: stream, median over seeds of G at B = 96, 48 and 192, of G relative to the arm the criterion selects (B = 96),
of the number of switches and of the fraction of readings on the learned arm (B = 96), number of seeds.
n_pass: streams with median G at B = 96 of at least 5% (the rule needs 3 of 11). Writes gate1_summary.json next to this
script; run from anywhere, for example in caps/adaptA:  python3 gate1_build.py"""
import json, glob, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "oracle_*_lightweight.json"))):
    J = json.load(open(f)); R = J["rows"]
    m = lambda key, b="B96": st.median(r[b][key] for r in R)
    rows.append([J["ds"], m("G_best"), m("G_best", "B48"), m("G_best", "B192"), m("G_crit"), m("switches"), m("fracL"), len(R)])
out = {"rows": rows, "n_pass": sum(r[1] >= 0.05 for r in rows)}
json.dump(out, open(os.path.join(HERE, "gate1_summary.json"), "w"), indent=1)
print("streams with median G96 >= 5%:", out["n_pass"], "of", len(rows))
