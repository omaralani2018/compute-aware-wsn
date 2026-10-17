"""Pre-registered analysis for PREREG15 (written before any radio run): energy of one report on a node's own radio.
Usage: python3 analyse_radio.py <serial.txt> <tx_consumption.oml> <tag> [<serial.txt again, for the RX counts>]
(amended by AMENDMENT_15a: TX lines selected by node name; validity from the RX node's per-variant counts)
The serial file has 'timestamp;node;message' lines (serial_aggregator); the consumption file has the 8 columns of the
IoT-LAB .oml consumption records (t_s and t_us in columns 4 and 5, power, voltage, current in 6 to 8)."""
import os, re, sys, json, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ser_f, cons_f, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rx_f = sys.argv[4] if len(sys.argv) > 4 else None
VARS = ["S_BC", "S_UC", "L_BC"]
TRIM = 0.300                     # seconds dropped at each end of every phase before averaging
ser = [l.rstrip("\n").split(";", 2) for l in open(ser_f, errors="replace") if l.count(";") >= 2]
TXNODE = next(n for _, n, m in ser if m.startswith("HDR") and "role=TX" in m)   # AMENDMENT_15a: TX node lines only
ser = [x for x in ser if x[1] == TXNODE]
boots = [i for i, (_, _, m) in enumerate(ser) if "This is RIOT" in m]
ser = ser[boots[-1]:] if boots else ser              # keep the run after the last boot
hdr = next(m for _, _, m in ser if m.startswith("HDR") and "role=TX" in m)
board = re.search(r"board=(\S+)", hdr).group(1); F = float(re.search(r"core_hz=(\d+)", hdr).group(1))
txline = next((m for _, _, m in ser if m.startswith("TX ")), "")
ph, open_ = [], {}
for ts, _, m in ser:
    g = re.match(r"B (IDLE|RUN (\S+)) (begin|end) (-?\d+)", m)
    if not g: continue
    key = (g.group(2) or "IDLE", int(g.group(4)))
    if g.group(3) == "begin": open_[key] = float(ts); continue
    d = dict(kind=key[0], round=key[1], t0=open_.pop(key), t1=float(ts))
    for f in ("calls", "cycles", "wall_ms", "tx_ok", "tx_fail", "send_err"):
        x = re.search(f + r"=(\d+)", m)
        if x: d[f] = int(x.group(1))
    ph.append(d)
D = np.array([[float(x) for x in l.split()] for l in open(cons_f) if len(l.split()) == 8])
t = D[:, 3] + D[:, 4] * 1e-6; P = D[:, 5]
# Alignment (fixed rule): the clock offset between the power monitor and the serial log is the shift (search -120 to
# 120 s, step 0.01 s) that maximises the correlation between the median-filtered power trace and the expected pattern (RUN phases
# on, IDLE phases off; the 12-s SYNC block makes the match unique). Every phase is then the serial interval shifted by that offset, trimmed by TRIM at both ends.
on = [p for p in ph if p["kind"] != "IDLE"]
lo_t, hi_t = min(p["t0"] for p in ph) - 3, max(p["t1"] for p in ph) + 3
grid = np.arange(lo_t, hi_t, 0.02)
pat = np.zeros(len(grid))
for p in on: pat[(grid >= p["t0"]) & (grid < p["t1"])] = 1
from scipy.ndimage import median_filter
Pm = median_filter(P, size=45, mode="nearest")      # 45-sample running median (about 0.2 s) against sampling noise
best = (-2.0, 0.0)
for off in np.arange(-120, 120, 0.01):
    pp = np.interp(grid + off, t, Pm, left=np.nan, right=np.nan)
    ok = ~np.isnan(pp)
    if ok.mean() < 0.90: continue                     # at least 90% of the expected pattern must lie inside the trace
    c = np.corrcoef(pat[ok], pp[ok])[0, 1]
    if c > best[0]: best = (float(c), float(off))
corr, offset = best
def mean_p(p):
    s = (t >= p["t0"] + offset + TRIM) & (t <= p["t1"] + offset - TRIM)
    assert s.sum() > 50, (p, int(s.sum())); return float(P[s].mean())
for p in ph: p["P"] = mean_p(p)
idle = [p for p in ph if p["kind"] == "IDLE"]; P_idle = st.median(p["P"] for p in idle)
RXV = {}
if rx_f:
    for l in open(rx_f, errors="replace"):
        g = re.search(r"RXV (\d) (\d) (\d+)", l)
        if g: RXV.setdefault(int(g.group(1)), {})[int(g.group(2))] = int(g.group(3))
res = dict(board=board, core_hz=F, tag=tag, offset_s=offset, correlation=corr, P_idle_W=P_idle, tx_setup=txline, variants={})
print("%s  alignment offset %.2f s, correlation %.3f  P_idle %.4f W (rounds %s)" % (
    board, offset, corr, P_idle, [round(p["P"], 4) for p in idle]))
for v in VARS:
    rs = [p for p in ph if p["kind"] == v]; assert len(rs) == 3, (v, len(rs))
    E_inc = [(p["P"] - P_idle) * (p["wall_ms"] / 1000) / p["calls"] for p in rs]
    E_gross = [p["P"] * (p["wall_ms"] / 1000) / p["calls"] for p in rs]
    calls = sum(p["calls"] for p in rs); fail = sum(p.get("tx_fail", 0) for p in rs); ok = sum(p.get("tx_ok", 0) for p in rs)
    err = sum(p.get("send_err", 0) for p in rs)
    # AMENDMENT_15a: the link-layer counters are not updated by this RIOT version (new netdev API), so validity uses the
    # frames counted by the RX node per variant and round (last RXV line of each), against the reports sent
    rx = RXV.get(VARS.index(v), {}); received = sum(rx.values()) if rx else None
    r = dict(E_inc_J=st.median(E_inc), E_inc_rounds=E_inc, E_gross_J=st.median(E_gross), P_run_W=[p["P"] for p in rs],
             reports=calls, received=received, send_err=err, t_report_ms=st.median(p["wall_ms"] / p["calls"] for p in rs),
             cycles_per_report=st.median(p["cycles"] / p["calls"] for p in rs),
             valid=(err == 0 and received is not None and received >= 0.99 * calls))
    res["variants"][v] = r
    print("%-5s E_inc %8.2f uJ (rounds %s)  E_gross %8.2f uJ  %.3f ms/report  reports %d received %s send_err %d  %s" % (
        v, r["E_inc_J"] * 1e6, [round(x * 1e6, 2) for x in E_inc], r["E_gross_J"] * 1e6, r["t_report_ms"], calls, received, err,
        "VALID" if r["valid"] else "INVALID"))
ls = [p for p in ph if p["kind"] == "LISTEN"]
res["listen_minus_idle_W"] = st.median(p["P"] - P_idle for p in ls) if ls else None
if ls: print("LISTEN  receive-on power above idle %.4f W" % res["listen_minus_idle_W"])
ok_align = corr >= 0.8
res["alignment_ok"] = ok_align
print("alignment %s (correlation >= 0.8 required)" % ("OK" if ok_align else "FAILED"))
json.dump(res, open(os.path.join(HERE, f"radio_{tag}_summary.json"), "w"), indent=1, default=float)
