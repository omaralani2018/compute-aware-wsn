"""Version 1.6. Figure 7: measured board power of one IoT-LAB M3 node over one round, (a) forecasts (run 454471),
(b) radio reports (run 455109). Reads only shipped files and the clock offsets found by the analyses.
Run from caps/iotlab/ with SOURCE_DATE_EPOCH=1791417600; writes fig_power_trace.png and .pdf."""
import json, re, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "font.size": 8, "axes.linewidth": 0.6,
                 "axes.edgecolor": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                 "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#4a3aa7",
     "ink": "#0b0b0b", "muted": "#8a8984"}

def trace(path):
    D = np.array([[float(x) for x in l.split()] for l in open(path) if len(l.split()) == 8 and l[:1].isdigit()])
    return D[:, 3] + D[:, 4] * 1e-6, D[:, 5] * 1e3          # time (s), power (mW)

# (a) forecasts: phases already on the power clock in the summary written by analyse_iotlab.py
S = json.load(open("iotlab_m3_summary.json"))
ph = [p for p in S["phases"] if p["round"] in (0, None) or p["kind"] == "IDLE"]
r0 = [p for p in S["phases"] if p["round"] == 0]
idle_a = st.median(p["P"] for p in S["phases"] if p["kind"] == "IDLE") * 1e3
ta, Pa = trace("data/m3_cons_cut2.txt")
A0 = min(p["t0"] for p in r0) - 1.9; A1 = max(p["t1"] for p in r0) + 1.0
NAME_A = {"persistence": "persistence", "expsm": "smoothing", "lin2": "extrap.\nK = 2", "lin12": "extrap.\nK = 12",
          "lightweight": "lightweight\n208 MACs", "cnn_lstm": "CNN-LSTM\n99,488 MACs"}
COL_A = {"persistence": C["muted"], "expsm": C["aqua"], "lin2": C["blue"], "lin12": C["blue"],
         "lightweight": C["orange"], "cnn_lstm": C["violet"]}

# (b) radio: phases from the transmitting node's serial log, shifted by the offset of the radio analysis
R = json.load(open("radio/radio_m3_summary.json")); off = R["offset_s"]; idle_b = R["P_idle_W"] * 1e3
ser = [l.rstrip("\n").split(";", 2) for l in open("radio/data/radio_m3a_serial.txt", errors="replace") if l.count(";") >= 2]
tx = next(n for _, n, m in ser if m.startswith("HDR") and "role=TX" in m)
ser = [x for x in ser if x[1] == tx]
boots = [i for i, (_, _, m) in enumerate(ser) if "This is RIOT" in m]; ser = ser[boots[-1]:] if boots else ser
phb, op = [], {}
for ts, _, m in ser:
    g = re.match(r"B (IDLE|RUN (\S+)) (begin|end) (-?\d+)", m)
    if not g: continue
    k = (g.group(2) or "IDLE", int(g.group(4)))
    if g.group(3) == "begin": op[k] = float(ts)
    else:
        c = re.search(r"calls=(\d+)", m)
        phb.append(dict(kind=k[0], round=k[1], t0=op.pop(k) + off, t1=float(ts) + off, calls=int(c.group(1)) if c else None))
rb0 = [p for p in phb if p["round"] == 0]
tb, Pb = trace("radio/data/radio_m3a_21.oml")
B0 = min(p["t0"] for p in rb0) - 0.5; B1 = max(p["t1"] for p in rb0) + 0.5
NAME_B = {"S_BC": "4 B\nbroadcast", "S_UC": "4 B\nacknowledged", "L_BC": "100 B\nbroadcast", "LISTEN": "receive\nmode"}
COL_B = {"S_BC": C["orange"], "S_UC": C["yellow"], "L_BC": C["violet"], "LISTEN": C["aqua"]}

fig, ax = plt.subplots(2, 1, figsize=(6.8, 4.2))
for a, t, P, T0, T1, phs, names, cols, idle in ((ax[0], ta, Pa, A0, A1, r0, NAME_A, COL_A, idle_a),
                                                  (ax[1], tb, Pb, B0, B1, rb0, NAME_B, COL_B, idle_b)):
    w = (t >= T0) & (t <= T1)
    for p in phs:
        if p["kind"] == "IDLE": continue
        a.axvspan(p["t0"] - T0, p["t1"] - T0, color=cols[p["kind"]], alpha=0.14, lw=0)
        a.text((p["t0"] + p["t1"]) / 2 - T0, 1.0, names[p["kind"]], transform=a.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=7)
        if p.get("calls"):
            a.text((p["t0"] + p["t1"]) / 2 - T0, 0.42, f"{p['calls']:,}", transform=a.get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=6.5, color=C["ink"])
    a.plot(t[w] - T0, P[w], color=C["ink"], lw=0.35)
    a.axhline(idle, color=C["muted"], lw=0.7, ls="--")
    a.text(T1 - T0, idle, " idle", va="center", ha="left", fontsize=7, color=C["muted"])
    a.set_xlim(0, T1 - T0); a.set_ylabel("board power (mW)")
ax[1].set_xlabel("time (s)")
for a, l in zip(ax, ("(a)", "(b)")): a.text(-0.08, 1.12, l, transform=a.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout(h_pad=2.2)
fig.savefig("fig_power_trace.png"); fig.savefig("fig_power_trace.pdf")
print("forecast round:", [(p["kind"], round(p["P"] * 1e3, 1)) for p in r0])
print("radio round:", [(p["kind"], round(p["t1"] - p["t0"], 2)) for p in rb0], "offset", off)
