"""Synthetic serial log and consumption trace in the firmware's sequence, for testing analyse_netx.py before any run.
Usage: python3 sim_netx.py <build> <outdir> [scale_light] [drop_frac] [seed]
True above-idle energy per round = E_pred x scale (scale_light for the lightweight arm, 1 otherwise); frames dropped at
random with drop_frac; power sampled every 4.5 ms with noise as in the PREREG15 traces; clock offset 30.0 s."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
build, out = sys.argv[1], sys.argv[2]; sl = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
drop = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0; rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 1)
PLAN = json.load(open(os.path.join(HERE, "netx_plan.json"))); EXP = json.load(open(os.path.join(HERE, "check/expected_decisions.json")))[build]
R = PLAN["R"]; STREAMS = list(PLAN["streams"])
PF, PC = 20, {"m3_float": 400, "m3_int": 120, "nrf": 60}[build]
P_IDLE, NOISE, P_LISTEN = {"nrf": (0.415, 0.002, 0.026)}.get(build, (0.0808, 0.0004, 0.040))
os.makedirs(out, exist_ok=True)
import re
hq = open(os.path.join(HERE, "app", "netx_models_q.h" if build == "m3_int" else "netx_models.h")).read()
def val(si, t):
    v = re.search(r"%s\[\d+\] = \{([^}]*)\}" % (("XQ%d" if build == "m3_int" else "X%d") % si), hq).group(1).split(",")[t]
    return (int(v) & 0xffffffff).to_bytes(4, "little").hex() if build == "m3_int" else np.float32(float(v)).tobytes().hex()
T0 = 1.8e9; t = T0; S = []; seg = []        # seg: (t0, t1, above-idle power)
TXN, RXN = "m3-21" if build != "nrf" else "nrf52840dk-5", "m3-22" if build != "nrf" else "nrf52840dk-6"
def tx(m, at=None): S.append((at if at is not None else t, TXN, m))
tx("main(): This is RIOT!"); t += 60
tx("HDR netx board=x role=TX core_hz=1 ch=22"); tx("TX build=%s R=%d passes=%d" % ("int" if build == "m3_int" else "float", R, 5 if build == "nrf" else 3))
def idle(p):
    global t
    tx("B IDLE begin %d t=0" % p); t += 5; tx("B IDLE end %d t=0" % p)
def sync(p):
    global t
    tx("B RUN SYNC begin %d t=0" % p); seg.append((t, t + 12, P_LISTEN)); t += 12; tx("B RUN SYNC end %d t=0" % p)
idle(-1); sync(-1); idle(-1)
NP = 5 if build == "nrf" else 3
for p in range(NP):
    for si, ds in enumerate(STREAMS):
        ch = PLAN["streams"][ds]["choice"][build]
        for arm in [ch["node_ref"], "lightweight", "cnn_lstm"]:
            key = f"{ds}:{arm}"; per = PC if arm == "cnn_lstm" else PF; wall = R * per / 1000
            E = PLAN["streams"][ds]["pred"][build][arm] * (sl if arm == "lightweight" else 1.0)
            tx("B RUN %s begin %d t=0 period_ms=%d" % (key, p, per)); seg.append((t, t + wall, E * R / wall))
            bits = bytes.fromhex(EXP[key]["bits"]); a = {"persistence": 0, "expsm": 1, "linear_extrap": 2, "lightweight": 3, "cnn_lstm": 4}[arm]
            for r in range(R):
                if (bits[r >> 3] >> (r & 7)) & 1 and rng.random() >= drop:
                    S.append((t + r * per / 1000 + 0.005, RXN, "F %d %d %d %s" % (si, a, r, val(si, 12 + r))))
            t += wall
            tx("B RUN %s end %d t=0 tx=%d send_err=0 late=0 fkcyc=1 akcyc=1 wall_ms=%d" % (key, p, EXP[key]["tx"], int(wall * 1000)))
            tx("D %s %d %s" % (key, p, EXP[key]["bits"])); idle(p)
sync(NP); idle(NP); tx("ALL DONE")
S.sort(key=lambda x: x[0])
if os.environ.get("SIM_SINK") == "v2":            # AMENDMENT_17b test: sink v2 lines every 10 s, with random corruption
    M = 0xffffffff
    def fm(h):
        h &= M; h ^= h >> 16; h = (h * 0x85ebca6b) & M; h ^= h >> 13; h = (h * 0xc2b2ae35) & M; h ^= h >> 16; return h
    pc = float(os.environ.get("SIM_CORRUPT", "0")); cnt = {}; hs = {}; out2 = []; nxt = T0 + 10
    def flush(at):
        for (s_, a_) in sorted(cnt):
            c, h = cnt[(s_, a_)], hs[(s_, a_)]; line = "RXV %d %d %d %08x %08x" % (s_, a_, c, h, fm(fm(fm(fm(s_) ^ a_) ^ c) ^ h))
            if rng.random() < pc:
                k = rng.integers(1, 4); idx = sorted(rng.choice(len(line), k, replace=False)); line = "".join(ch for i, ch in enumerate(line) if i not in idx)
            out2.append((at, RXN, line))
    for x in S:
        while x[0] >= nxt: flush(nxt); nxt += 10
        if x[1] == RXN and x[2].startswith("F "):
            _, s_, a_, r_, v_ = x[2].split(); s_, a_, r_ = int(s_), int(a_), int(r_)
            cnt[(s_, a_)] = cnt.get((s_, a_), 0) + 1
            hs[(s_, a_)] = (hs.get((s_, a_), 0) + fm(((s_ << 28) | (a_ << 24) | r_) ^ fm(int.from_bytes(bytes.fromhex(v_), "little")))) & M
        else: out2.append(x)
    for _ in range(5): flush(nxt); nxt += 10
    S = sorted(out2, key=lambda x: x[0])
open(os.path.join(out, "serial.txt"), "w").write("".join("%.6f;%s;%s\n" % x for x in S))
OFF = 30.0; ts = np.arange(T0 + OFF - 5, t + OFF + 5, 0.0045); P = P_IDLE + rng.normal(0, NOISE, len(ts))
DRIFT = float(os.environ.get("SIM_DRIFT", "0"))          # seconds of offset change from start to end
for a, b, w in seg:
    o = OFF + DRIFT * (a - T0) / (t - T0); P[(ts >= a + o) & (ts < b + o)] += w
with open(os.path.join(out, "sender.oml"), "w") as f:
    f.write("protocol: 5\ncontent: text\n\n")
    for i, (x, y) in enumerate(zip(ts, P)): f.write("%.6f\t1\t%d\t%d\t%d\t%.6f\t4.9\t0.08\n" % (x - T0, i + 1, int(x), int(round((x - int(x)) * 1e6)) % 1000000, y))
print("simulated", build, "duration %.1f min" % ((t - T0) / 60))
