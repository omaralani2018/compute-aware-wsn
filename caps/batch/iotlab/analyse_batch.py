"""PREREG16 analysis (written before any run). Usage:
python3 analyse_batch.py <serial.txt> <tx_consumption.oml> <tag> <sched.json> <PREREG15 radio summary json>
Serial: 'timestamp;node;message' (serial_aggregator). Consumption: IoT-LAB .oml, 8 columns (t_s, t_us in 4-5; power in 6)."""
import os, re, sys, json, statistics as st
import numpy as np
from scipy.ndimage import median_filter
ser_f, cons_f, tag, sched_f, r15_f = sys.argv[1:6]
SCH = json.load(open(sched_f)); R15 = json.load(open(r15_f))["variants"]
TRIM = 0.300
ser = [l.rstrip("\n").split(";", 2) for l in open(ser_f, errors="replace") if l.count(";") >= 2]
TX = next(n for _, n, m in ser if m.startswith("HDR") and "role=TX" in m)
RXL = [m for _, n, m in ser if n != TX]
ser = [x for x in ser if x[1] == TX]
boots = [i for i, (_, _, m) in enumerate(ser) if "This is RIOT" in m]
ser = ser[boots[-1]:] if boots else ser
hdr = next(m for _, _, m in ser if m.startswith("HDR") and "role=TX" in m)
board = re.search(r"board=(\S+)", hdr).group(1)
ph, open_ = [], {}
for ts, _, m in ser:
    g = re.match(r"B (IDLE|RUN (\S+)) (begin|end) (-?\d+)", m)
    if not g: continue
    key = (g.group(2) or "IDLE", int(g.group(4)))
    if g.group(3) == "begin": open_[key] = float(ts); continue
    d = dict(kind=key[0], round=key[1], t0=open_.pop(key), t1=float(ts))
    for f in ("calls", "reads", "wall_ms", "send_err", "late"):
        x = re.search(f + r"=(\d+)", m)
        if x: d[f] = int(x.group(1))
    ph.append(d)
def _num(l):
    try: return [float(x) for x in l.split()]
    except ValueError: return None
_rows = [_num(l) for l in open(cons_f, errors="replace") if len(l.split()) == 8 and l[:1].isdigit()]
SKIPPED = sum(r is None for r in _rows)
D = np.array([r for r in _rows if r is not None])
t = D[:, 3] + D[:, 4] * 1e-6; P = D[:, 5]
# alignment (as PREREG15): SYNC and sweep phases on, IDLE off; replay phases masked out of the correlation
lo_t, hi_t = max(min(p["t0"] for p in ph) - 3, t.min() + 305), min(max(p["t1"] for p in ph) + 3, t.max() - 305)
grid = np.arange(lo_t, hi_t, 0.02); pat = np.zeros(len(grid)); mask = np.ones(len(grid), bool)
for p in ph:
    s = (grid >= p["t0"]) & (grid < p["t1"])
    if p["kind"].startswith("R_"): mask[s] = False
    elif p["kind"] != "IDLE": pat[s] = 1
Pm = median_filter(P, size=45, mode="nearest"); best = (-2.0, 0.0)
for off in np.arange(-300, 300, 0.01):
    pp = np.interp(grid + off, t, Pm, left=np.nan, right=np.nan); ok = ~np.isnan(pp) & mask
    if ok.sum() < 0.90 * mask.sum(): continue
    c = np.corrcoef(pat[ok], pp[ok])[0, 1]
    if c > best[0]: best = (float(c), float(off))
corr, offset = best
def mean_p(p):
    a, b = p["t0"] + offset + TRIM, p["t1"] + offset - TRIM
    if a < t.min() or b > t.max(): return None          # phase not covered by the power trace: not used
    s = (t >= a) & (t <= b); assert s.sum() > 50, (p, int(s.sum())); return float(P[s].mean())
for p in ph: p["P"] = mean_p(p)
uncovered = [(p["kind"], p["round"]) for p in ph if p["P"] is None]
ph = [p for p in ph if p["P"] is not None]
P_idle = st.median(p["P"] for p in ph if p["kind"] == "IDLE")
RXV = {}
for m in RXL:
    g = re.search(r"RXV (\d+) (\d+) (\d+)", m)
    if g: RXV.setdefault(int(g.group(1)), {})[int(g.group(2))] = int(g.group(3))
node = "M3" if "m3" in board else "nRF52840"
res = dict(skipped_sample_lines=SKIPPED, rx_counts_available=bool(RXV), uncovered_phases=uncovered, trace_start=float(t.min()), board=board, tag=tag, offset_s=offset, correlation=corr, alignment_ok=corr >= 0.8, P_idle_W=P_idle, sweep={}, cells={})
# sweep: energy of one frame per payload size
sizes = [4, 20, 40, 60, 80, 100]
for i, b in enumerate(sizes):
    rs = [p for p in ph if p["kind"] == f"SW{b}"]
    E = [(p["P"] - P_idle) * p["wall_ms"] / 1000 / p["calls"] for p in rs]
    calls = sum(p["calls"] for p in rs); rec = sum(RXV.get(i, {}).values())
    res["sweep"][b] = dict(E_inc_J=st.median(E), rounds=E, frames=calls, received=rec, valid=(rec >= 0.99 * calls if RXV else None) is not False and sum(p.get("send_err", 0) for p in rs) == 0, rx_count_available=bool(RXV))
e4, e100 = res["sweep"][4]["E_inc_J"], res["sweep"][100]["E_inc_J"]
line = lambda b: e4 + (e100 - e4) * (b - 4) / 96
dev = {b: res["sweep"][b]["E_inc_J"] / line(b) - 1 for b in (20, 40, 60, 80)}
res["P1_linearity_max_dev"] = max(abs(v) for v in dev.values()); res["P1_pass"] = res["P1_linearity_max_dev"] <= 0.10; res["P1_dev"] = dev
res["vs_PREREG15"] = dict(S_BC_4B=(e4, R15["S_BC"]["E_inc_J"]), L_BC_100B=(e100, R15["L_BC"]["E_inc_J"]))
# replay: energy per reading above idle, minus the NULL cell (CPU wake-ups common to every scheme)
def per_round(name):
    return {p["round"]: (p["P"] - P_idle) * p["wall_ms"] / 1000 / p["reads"] for p in ph if p["kind"] == "R_" + name}
def per_read(name):                          # net of the NULL cell of the same round, median over rounds
    c, z = per_round(name), per_round("NULL")
    return st.median(c[r] - z[r] for r in c if r in z), [p for p in ph if p["kind"] == "R_" + name]
e_null = st.median(per_round("NULL").values()); res["E_null_per_reading_J"] = e_null
names = list(SCH["cells"]); n_ok = 0
for j, nm in enumerate(names):
    net, rs = per_read(nm); e = net + e_null
    c = SCH["cells"][nm]
    pred15 = SCH["predicted_E_inc_per_reading_J"][nm][node]
    pred_run = None  # prediction from this run's own sweep line, computed from the schedule's frame sizes
    _c, _z = per_round(nm), per_round("NULL")
    res["cells"][nm] = dict(rounds_net={r: _c[r] - _z[r] for r in _c if r in _z}, E_net_per_reading_J=net, E_gross_inc_J=e, pred_PREREG15_J=pred15,
                            frames=sum(p["calls"] for p in rs), received=sum(RXV.get(6 + j, {}).values()),
                            late=sum(p.get("late", 0) for p in rs), send_err=sum(p.get("send_err", 0) for p in rs))
# P3 uses the run's own sweep line with the exact frame sizes of each schedule
FR = json.load(open(sched_f)).get("frame_sizes", {})
for nm, r in res["cells"].items():
    if nm in FR:
        r["pred_run_J"] = sum(line(b) for b in FR[nm]) / SCH["cells"][nm]["readings"]
        r["within_15pct"] = abs(r["E_net_per_reading_J"] / r["pred_run_J"] - 1) <= 0.15; n_ok += r["within_15pct"]
res["P3_cells_within_15pct"] = n_ok
# P2: ranking of the three schemes per stream as predicted
rk = {}
for s in SCH["streams"]:
    meas = {k: res["cells"][f"{s}_{k}"]["E_net_per_reading_J"] for k in ("S0", "S1", "S2")}
    pred = {k: SCH["predicted_E_inc_per_reading_J"][f"{s}_{k}"][node] for k in ("S0", "S1", "S2")}
    rk[s] = dict(measured=sorted(meas, key=meas.get), predicted=sorted(pred, key=pred.get))
    rk[s]["same"] = rk[s]["measured"] == rk[s]["predicted"]
res["P2_rankings"] = rk; res["P2_pass"] = all(v["same"] for v in rk.values())
print(json.dumps({k: res[k] for k in ("board", "correlation", "alignment_ok", "P1_linearity_max_dev", "P1_pass", "P2_pass", "P3_cells_within_15pct")}, default=float))
for b, s in res["sweep"].items(): print("SW%-4d E %.3e J  valid=%s" % (b, s["E_inc_J"], s["valid"]))
for nm, r in res["cells"].items(): print("%-16s meas %.3e  pred(run) %.3e  pred(15) %.3e  late=%d" % (nm, r["E_net_per_reading_J"], r.get("pred_run_J", float("nan")), r["pred_PREREG15_J"], r["late"]))
json.dump(res, open(f"batch_{tag}_summary.json", "w"), indent=1, default=float)
