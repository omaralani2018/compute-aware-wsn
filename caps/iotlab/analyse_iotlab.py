"""Pre-registered analysis for PREREG12 (written before the power file was inspected).
Usage: python3 analyse_iotlab.py <serial.txt> <consumption_cut.txt> <tag>"""
import os, re, sys, json, struct, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
ser_f, cons_f, tag = sys.argv[1], sys.argv[2], sys.argv[3]
NAMES = ["persistence", "expsm", "lin2", "lin12", "lightweight", "cnn_lstm"]
MACS = {"persistence": 0, "expsm": 12, "lin2": 2, "lin12": 12, "lightweight": 208, "cnn_lstm": 99488}
TRIM = 0.300
ser = [l.rstrip("\n").split(";", 2) for l in open(ser_f, errors="replace") if l.count(";") >= 2]
# keep only the run after the last boot (the capture can start with buffered lines of the run before the reset)
ser = ser[max(i for i, (_, _, m) in enumerate(ser) if "This is RIOT" in m):]
hdr = next(m for _, _, m in ser if m.startswith("HDR"))
F = float(re.search(r"core_hz=(\d+)", hdr).group(1)); board = re.search(r"board=(\S+)", hdr).group(1)
ref = json.load(open(os.path.join(HERE, "ref.json")))
res = {}
# Part A: correctness and cycles
for k in NAMES:
    rows = [m.split() for _, _, m in ser if m.startswith("A " + k + " ")]
    assert len(rows) == 100, (k, len(rows))
    c = np.array([int(r[2]) for r in rows], float)
    out = np.array([struct.unpack(">f", bytes.fromhex(r[3]))[0] for r in rows])
    rr = np.array(ref[k]); rel = float(np.max(np.abs(out - rr) / np.maximum(np.abs(rr), 1e-12)))
    res[k] = dict(cycles_mean=float(c.mean()), cycles_max=int(c.max()), max_rel_err=rel, ok=rel < 1e-4, t_s=float(c.mean() / F))
# Part B: phases from serial timestamps
ph = []; open_ = {}
for ts, _, m in ser:
    g = re.match(r"B (IDLE|RUN (\S+)) (begin|end) (\d+)", m)
    if not g: continue
    key = (g.group(2) or "IDLE", int(g.group(4)))
    if g.group(3) == "begin": open_[key] = float(ts)
    else:
        b = re.search(r"calls=(\d+) cycles=(\d+)", m)
        ph.append(dict(kind=key[0], round=key[1], t0=open_.pop(key), t1=float(ts),
                       calls=int(b.group(1)) if b else None, cycles=int(b.group(2)) if b else None))
D = np.array([[float(x) for x in l.split()] for l in open(cons_f) if len(l.split()) == 8])
t = D[:, 3] + D[:, 4] * 1e-6; P = D[:, 5]; V = D[:, 6]; I = D[:, 7]
def mean_p(a, b):
    s = (t >= a + TRIM) & (t <= b - TRIM); assert s.sum() > 50, (a, b, s.sum()); return float(P[s].mean()), int(s.sum())
# Pre-registered fallback: the serial and power clocks disagree, so phases are located on the power trace as the
# contiguous intervals above the midpoint between the idle and run levels, matched in time order to the known sequence.
# Matching: the clock offset is the shift (search -120..120 s, step 0.01 s) that maximises the correlation between the
# power trace and the expected on/off pattern from the serial log (Part A and the 18 RUN phases on, IDLE off); levels are
# the medians of the trace inside the shifted IDLE and RUN phases; each expected phase takes the above-midpoint interval
# (longer than 1 s) that overlaps it most. IDLE phases are the gaps between consecutive matched intervals.
runs_ser = [p for p in ph if p["kind"] != "IDLE"]; idles_ser = [p for p in ph if p["kind"] == "IDLE"]
partA = dict(kind="PARTA", t0=float(next(ts for ts, _, m in ser if m.startswith("A persistence"))),
             t1=float(next(ts for ts, _, m in ser if m == "A done")))
on = [partA] + runs_ser
lo_t, hi_t = min(p["t0"] for p in on) - 3, max(p["t1"] for p in ph) + 3
grid = np.arange(lo_t, hi_t, 0.02)
def pattern(g):
    y = np.zeros(len(g))
    for p in on: y[(g >= p["t0"]) & (g < p["t1"])] = 1
    return y
pat = pattern(grid); best = (-2, 0)
for off in np.arange(-120, 120, 0.01):
    pp = np.interp(grid + off, t, P, left=np.nan, right=np.nan)
    if np.isnan(pp).mean() > 0.01: continue
    c = np.corrcoef(pat, pp)[0, 1]
    if c > best[0]: best = (c, off)
corr, offset = best
def med_in(ps):
    v = np.concatenate([P[(t >= p["t0"] + offset + TRIM) & (t <= p["t1"] + offset - TRIM)] for p in ps])
    return float(np.median(v))
lo, hi = med_in(idles_ser), med_in(runs_ser); mid = (lo + hi) / 2
# the on/off mask uses a 45-sample (about 0.2 s) running median of the trace, because single samples are noisy on
# boards with a small run-idle contrast; phase means are still taken on the raw samples, trimmed by 300 ms
from scipy.ndimage import median_filter
Pm = median_filter(P, size=45, mode="nearest")
w = (t >= lo_t + offset) & (t <= hi_t + offset); tw, above = t[w], Pm[w] > mid
edges = np.flatnonzero(np.diff(above.astype(int)))
starts = [tw[i + 1] for i in edges if above[i + 1]]; ends = [tw[i + 1] for i in edges if not above[i + 1]]
if above[0]: starts = [tw[0]] + starts
iv = []
for s0 in starts:
    e = [x for x in ends if x > s0]
    if e and e[0] - s0 > 1.0: iv.append((s0, e[0]))
def ov(a, b): return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
match = [max(range(len(iv)), key=lambda j: ov(iv[j], (p["t0"] + offset, p["t1"] + offset))) for p in on]
assert len(set(match)) == len(on) and match == sorted(match), match
print("alignment: offset (power minus serial) %.3f s, correlation %.3f; levels idle %.4f W, run %.4f W, midpoint %.4f W; %d intervals > 1 s, %d matched" % (
    offset, corr, lo, hi, mid, len(iv), len(on)))
for j, p in zip(match[1:], runs_ser): p["t0"], p["t1"] = iv[j]
mi = [iv[j] for j in match]
idle_new = [dict(kind="IDLE", round=None, t0=mi[j][1], t1=mi[j + 1][0], calls=None, cycles=None) for j in range(len(mi) - 1)]
ph = runs_ser + idle_new
for p in ph: p["P"], p["n"] = mean_p(p["t0"], p["t1"])
# alignment check: the power level must change at the phase edges (pre-registered fallback otherwise)
idle = [p for p in ph if p["kind"] == "IDLE"]; run = [p for p in ph if p["kind"] != "IDLE"]
P_idle = st.median(p["P"] for p in idle)
edge = []
for p in run:
    s_in = (t >= p["t0"] + 0.05) & (t <= p["t0"] + 0.25); s_out = (t >= p["t0"] - 0.25) & (t <= p["t0"] - 0.05)
    edge.append(float(P[s_in].mean() - P[s_out].mean()))
for k in NAMES:
    rs = [p for p in run if p["kind"] == k]; assert len(rs) == 3, (k, len(rs))
    Pr = st.median(p["P"] for p in rs); tB = st.median(p["cycles"] / p["calls"] / F for p in rs)
    tk = res[k]["t_s"]
    res[k].update(P_run_W=Pr, P_idle_W=P_idle, t_partB_s=tB, E_inc_J=(Pr - P_idle) * tk, E_gross_J=Pr * tk,
                  nJ_per_MAC=((Pr - P_idle) * tk * 1e9 / MACS[k]) if MACS[k] else None,
                  P_run_rounds=[p["P"] for p in rs])
print("%s  core %.0f MHz  P_idle %.4f W (rounds %s)  V mean %.3f" % (board, F / 1e6, P_idle, [round(p["P"], 4) for p in idle], V.mean()))
for k in NAMES:
    r = res[k]
    print("%-12s cycles %11.0f  t %10.4f ms (partB %.4f)  P_run %.4f W  E_inc %10.3f uJ  E_gross %10.3f uJ  %s nJ/MAC  err %.1e %s" % (
        k, r["cycles_mean"], r["t_s"] * 1e3, r["t_partB_s"] * 1e3, r["P_run_W"], r["E_inc_J"] * 1e6, r["E_gross_J"] * 1e6,
        "%.2f" % r["nJ_per_MAC"] if r["nJ_per_MAC"] else "-", r["max_rel_err"], "OK" if r["ok"] else "MISMATCH"))
print("edge check (run minus idle, first 0.2 s of each RUN): min %.4f W, median %.4f W" % (min(edge), st.median(edge)))
# CAPS with the measured incremental energies, exactly as analyse_avr.py
ER = 3.8e-4; MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
def e_arm(name, K=None):
    if name == "persistence": return res["persistence"]["E_inc_J"]
    if name == "expsm": return res["expsm"]["E_inc_J"]
    return res["lin2"]["E_inc_J"] + (K - 2) / 10 * (res["lin12"]["E_inc_J"] - res["lin2"]["E_inc_J"])
out = {"board": board, "core_hz": F, "per_forecast": res, "phases": ph, "edge": edge, "streams": {}}
sav = {"lightweight": 0, "cnn_lstm": 0}; scr = {"lightweight": 0, "cnn_lstm": 0}; wrong = []
for ds, d in MS.items():
    bt = d["by_tau"]["0.1"]; arms = ["persistence", "expsm", "linear_extrap"]
    E = {a: e_arm(a, bt["K"]) for a in arms}
    sel = min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + E[a], arms.index(a)))
    tR = bt["test"][sel]["rho"]; vR = bt["val"][sel]
    L = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
    row = dict(reference=sel, reference_paper=bt["selected"], E_ref_uJ=E[sel] * 1e6)
    for k in ("lightweight", "cnn_lstm"):
        Ek = res[k]["E_inc_J"]
        net = [(e["test_rho"] - tR) * ER - (Ek - E[sel]) for e in L if e["predictor"] == k]
        s = (Ek - E[sel]) >= (1 - vR) * ER; m = st.median(net)
        row[k] = dict(median_net_uJ=m * 1e6, seeds_positive=sum(x > 0 for x in net), screened=bool(s), correct=(not s) or m <= 0)
        sav[k] += m > 0; scr[k] += s
        if s and m > 0: wrong.append((ds, k))
    out["streams"][ds] = row
    print("%-17s ref %-13s (paper %-13s) | lightweight net %+9.2f uJ (%2d/20 >0) screened %-5s | CNN-LSTM net %+10.1f uJ screened %s" % (
        ds, sel, bt["selected"], row["lightweight"]["median_net_uJ"], row["lightweight"]["seeds_positive"], row["lightweight"]["screened"],
        row["cnn_lstm"]["median_net_uJ"], row["cnn_lstm"]["screened"]))
print("positive median net saving: lightweight %d/11, CNN-LSTM %d/11 | screened: lightweight %d/11, CNN-LSTM %d/11 | false exclusions: %s" % (
    sav["lightweight"], sav["cnn_lstm"], scr["lightweight"], scr["cnn_lstm"], wrong or "none"))
out.update(n_positive=sav, n_screened=scr, false_exclusions=wrong)
json.dump(out, open(os.path.join(HERE, f"iotlab_{tag}_summary.json"), "w"), indent=1, default=float)
