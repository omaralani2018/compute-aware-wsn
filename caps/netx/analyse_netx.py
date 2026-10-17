"""Pre-registered analysis for PREREG17 (NETX), written and tested on synthetic traces before any run.
Usage: python3 analyse_netx.py <build: m3_float|m3_int|nrf> <serial.txt> <sender_consumption.oml> <tag>
The serial file holds the lines of both nodes ('timestamp;node;message', serial_aggregator); the consumption file is the
sender's IoT-LAB .oml record (t_s, t_us in columns 4 and 5, power in column 6)."""
import os, re, sys, json, statistics as st
import numpy as np
from scipy.ndimage import median_filter
HERE = os.path.dirname(os.path.abspath(__file__))
build, ser_f, cons_f, tag = sys.argv[1:5]
PLAN = json.load(open(os.path.join(HERE, "netx_plan.json"))); HOST = json.load(open(os.path.join(HERE, "netx_host.json")))
EXP = json.load(open(os.path.join(HERE, "check/expected_decisions.json")))[build]
R = PLAN["R"]; STREAMS = list(PLAN["streams"]); TRIM = 0.300
ser = [l.rstrip("\n").split(";", 2) for l in open(ser_f, errors="replace") if l.count(";") >= 2]
TX = next(n for _, n, m in ser if m.startswith("HDR netx") and "role=TX" in m)
RXN = next((n for _, n, m in ser if m.startswith("HDR netx") and "role=RX" in m), None) or next((n for _, n, m in ser if m.startswith("F ")), None)
txl = [x for x in ser if x[1] == TX]
boots = [i for i, (_, _, m) in enumerate(txl) if "This is RIOT" in m]
txl = txl[boots[-1]:] if boots else txl
t_first = float(txl[0][0])
hdr = next(m for _, _, m in txl if m.startswith("TX build="))
PASSES = int(re.search(r"passes=(\d+)", hdr).group(1))
assert re.search(r"build=(\S+)", hdr).group(1) == ("int" if build == "m3_int" else "float"), hdr
# ---- phases of the sender
ph, open_ = [], {}
for ts, _, m in txl:
    g = re.match(r"B (IDLE|RUN (\S+)) (begin|end) (-?\d+)", m)
    if g:
        key = (g.group(2) or "IDLE", int(g.group(4)))
        if g.group(3) == "begin": open_[key] = float(ts); continue
        d = dict(kind=key[0], p=key[1], t0=open_.pop(key), t1=float(ts))
        for f in ("tx", "send_err", "late", "fkcyc", "akcyc", "wall_ms"):
            x = re.search(f + r"=(\d+)", m)
            if x: d[f] = int(x.group(1))
        ph.append(d); continue
    g = re.match(r"D (\S+) (-?\d+) ([0-9a-f]+)", m)
    if g:
        last = [q for q in ph if q["kind"] == g.group(1) and q["p"] == int(g.group(2))][-1]; last["bits"] = g.group(3)
for i, q in enumerate(ph): q["i"] = i
# ---- frames received by the sink, assigned to the sender's RUN phases by stream, arm and order of the round index
ARMID = {"persistence": 0, "expsm": 1, "linear_extrap": 2, "lightweight": 3, "cnn_lstm": 4}
frames = [(float(ts), m.split()) for ts, n, m in ser if n == RXN and m.startswith("F ") and len(m.split()) == 5 and float(ts) >= t_first]
runs = [q for q in ph if q["kind"] not in ("IDLE", "SYNC")]
for q in runs:
    ds, arm = q["kind"].split(":"); s, a = STREAMS.index(ds), ARMID[arm]
    q["rx"] = {int(f[3]): f[4] for ts, f in frames if int(f[1]) == s and int(f[2]) == a and q["t0"] - 1 <= ts <= q["t1"] + 3}
# ---- alignment (fixed rule): offset in [-120, 120] s, step 0.01 s, maximising the correlation between the 45-sample
# running median of power and the pattern SYNC on / adjacent IDLE off, over the two SYNC blocks with their 5-s IDLE neighbours
D = np.array([[float(x) for x in l.split()] for l in open(cons_f) if len(l.split()) == 8 and l[:1].isdigit()])
t = D[:, 3] + D[:, 4] * 1e-6; P = D[:, 5]; Pm = median_filter(P, size=45, mode="nearest")
syncs = [q for q in ph if q["kind"] == "SYNC"]; assert len(syncs) == 2, len(syncs)
def best_off(blocks):
    g = np.concatenate([np.arange(b["t0"] - 5, b["t1"] + 5, 0.02) for b in blocks])
    pat = np.zeros(len(g))
    for b in blocks: pat[(g >= b["t0"]) & (g < b["t1"])] = 1
    best = (-2.0, 0.0)
    for off in np.arange(-120, 120, 0.01):
        pp = np.interp(g + off, t, Pm, left=np.nan, right=np.nan); ok = ~np.isnan(pp)
        if ok.mean() < 0.90: continue
        c = np.corrcoef(pat[ok], pp[ok])[0, 1]
        if c > best[0]: best = (float(c), float(off))
    return best
corr, offset = best_off(syncs); (c1, o1), (c2, o2) = best_off(syncs[:1]), best_off(syncs[1:])
align_ok = corr >= 0.9 and abs(o1 - o2) <= 0.1
def mean_p(q):
    s = (t >= q["t0"] + offset + TRIM) & (t <= q["t1"] + offset - TRIM)
    return float(P[s].mean()) if s.sum() > 50 else float("nan")
for q in ph: q["P"] = mean_p(q)
# ---- per RUN: validity, decisions, energy above the adjacent IDLE phases
for q in runs:
    ds, arm = q["kind"].split(":")
    want = EXP[q["kind"]]["bits"]; q["dec_ok"] = q.get("bits") == want
    sent = {r for r in range(R) if (bytes.fromhex(q.get("bits", want))[r >> 3] >> (r & 7)) & 1}
    q["received"] = len(set(q["rx"]) & sent); q["extra"] = len(set(q["rx"]) - sent)
    q["valid"] = q.get("send_err", 1) == 0 and q.get("tx", 0) > 0 and q["received"] >= 0.99 * q["tx"] and q.get("wall_ms") is not None
    nb = [ph[q["i"] - 1], ph[q["i"] + 1]] if 0 < q["i"] < len(ph) - 1 else []
    nb = [b for b in nb if b["kind"] == "IDLE"]
    q["P_idle"] = float(np.mean([b["P"] for b in nb])) if len(nb) == 2 else float("nan")
    q["E_meas"] = (q["P"] - q["P_idle"]) * q["wall_ms"] / 1000 / R if q.get("wall_ms") else float("nan")
    q["E_gross"] = q["P"] * q["wall_ms"] / 1000 / R if q.get("wall_ms") else float("nan")
    q["valid"] = q["valid"] and bool(np.isfinite(q["E_meas"]))       # a gap in the power trace gives no measurement
# received readings equal the readings the sender holds (float bits or Q(fi) integers), from the generator's data
def readings():
    hq = open(os.path.join(HERE, "app", "netx_models_q.h" if build == "m3_int" else "netx_models.h")).read()
    out = {}
    for si, ds in enumerate(STREAMS):
        nm = ("XQ%d" if build == "m3_int" else "X%d") % si
        body = re.search(r"%s\[\d+\] = \{([^}]*)\}" % nm, hq).group(1).split(",")
        if build == "m3_int": out[ds] = [(int(v) & 0xffffffff).to_bytes(4, "little").hex() for v in body]
        else: out[ds] = [np.float32(float(v)).tobytes().hex() for v in body]
    return out
XR = readings()
for q in runs:
    ds = q["kind"].split(":")[0]; q["values_ok"] = all(XR[ds][12 + r] == v for r, v in q["rx"].items())
# ---- per arm: median over valid passes
pred = PLAN["streams"]; res = dict(build=build, tag=tag, tx_node=TX, rx_node=RXN, offset_s=offset, correlation=corr,
                                    offset_sync_start=o1, offset_sync_end=o2, alignment_ok=align_ok, cells={})
EREP = {"m3_float": "radio_m3_summary.json", "m3_int": "radio_m3_summary.json", "nrf": "radio_nrf_summary.json"}
erep = json.load(open(os.path.join(HERE, "..", "iotlab", "radio", EREP[build])))["variants"]["S_UC"]["E_inc_J"]
o1n = o2n = o3n = cells_valid = 0; o2_tot = 0
for ds in STREAMS:
    ch = pred[ds]["choice"][build]; arms = [ch["node_ref"], "lightweight", "cnn_lstm"]; cell = {"arms": {}}
    for arm in arms:
        rs = [q for q in runs if q["kind"] == f"{ds}:{arm}"]; vr = [q for q in rs if q["valid"]]
        Ef = pred[ds]["pred"][build][arm] - EXP[f"{ds}:{arm}"]["tx"] / R * erep     # forecast part of the prediction
        ntx = st.median(q["tx"] for q in rs) if rs else None
        Ep = Ef + (ntx if ntx is not None else 0) / R * erep                           # n_tx from the node's own transmissions
        cell["arms"][arm] = dict(E_meas=st.median(q["E_meas"] for q in vr) if len(vr) >= max(2, (PASSES + 1) // 2) else None,
                                 E_meas_passes=[q["E_meas"] for q in rs], E_gross=st.median(q["E_gross"] for q in vr) if len(vr) >= max(2, (PASSES + 1) // 2) else None,
                                 E_pred=Ep, tx=[q["tx"] for q in rs], received=[q["received"] for q in rs], extra=[q["extra"] for q in rs],
                                 send_err=[q.get("send_err") for q in rs], late=[q.get("late") for q in rs], dec_ok=[q["dec_ok"] for q in rs],
                                 values_ok=[q["values_ok"] for q in rs], valid=[q["valid"] for q in rs], passes=len(rs))
    A = cell["arms"]; ok = align_ok and all(v["E_meas"] is not None for v in A.values())
    cell["valid"] = ok
    if ok:
        cells_valid += 1
        ep_min = min(v["E_pred"] for v in A.values()); m_arm = min(A, key=lambda k: A[k]["E_meas"])
        cell["O1"] = A[m_arm]["E_pred"] <= 1.05 * ep_min   # predicted ties (within 5% of the lowest) count as agreement
        cell["O2"] = {k: abs(v["E_meas"] / v["E_pred"] - 1) <= 0.15 for k, v in A.items()}
        em = min(v["E_meas"] for v in A.values()); cell["O3"] = A[ch["caps"]]["E_meas"] <= 1.05 * em
        cell["lowest_meas"] = m_arm; cell["caps"] = ch["caps"]; cell["blind"] = ch["blind"]
        o1n += cell["O1"]; o2n += sum(cell["O2"].values()); o2_tot += len(cell["O2"]); o3n += cell["O3"]
    res["cells"][ds] = cell
O4 = dict(decisions_equal=sum(all(v["dec_ok"]) for c in res["cells"].values() for v in c["arms"].values()),
          values_equal=sum(all(v["values_ok"]) for c in res["cells"].values() for v in c["arms"].values()),
          runs_valid=sum(q["valid"] for q in runs), runs=len(runs))
res.update(O1=[o1n, cells_valid], O2=[o2n, o2_tot], O3=[o3n, cells_valid], O4=O4)
vc = [c for c in res["cells"].values() if c["valid"]]
if vc:
    res["blind_vs_caps"] = sum(c["arms"][c["blind"]]["E_meas"] for c in vc) / sum(c["arms"][c["caps"]]["E_meas"] for c in vc) - 1
json.dump(res, open(os.path.join(HERE, f"netx_{tag}_summary.json"), "w"), indent=1, default=float)
print("%s %s  alignment offset %.2f s corr %.3f (start %.2f, end %.2f)  %s" % (build, tag, offset, corr, o1, o2, "OK" if align_ok else "FAILED"))
for ds, c in res["cells"].items():
    print("  %-13s %s" % (ds, "  ".join("%s %s/%s uJ tx %s rx %s%s" % (k, "%.1f" % (v["E_meas"] * 1e6) if v["E_meas"] is not None else "-",
          "%.1f" % (v["E_pred"] * 1e6), v["tx"], v["received"], "" if all(v["valid"]) else " INVALID") for k, v in c["arms"].items())),
          "" if not c["valid"] else "| O1 %s O3 %s (CAPS %s, lowest %s)" % (c["O1"], c["O3"], c["caps"], c["lowest_meas"]))
print("O1 %d/%d  O2 %d/%d  O3 %d/%d  O4 %s" % (o1n, cells_valid, o2n, o2_tot, o3n, cells_valid, O4))
if "blind_vs_caps" in res: print("compute-blind vs CAPS (measured, valid cells): %+.1f%%" % (100 * res["blind_vs_caps"]))
