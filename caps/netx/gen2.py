"""NETX generator, part 2 (after gen.py): firmware tables and the integer build.
 - app/netx_table.h : per-stream table for the float builds (pointers into netx_models.h, reference arm id)
 - app/netx_models_q.h : per-stream integer data and weights for the M3 integer build (PREREG13 scheme, qemu.py)
 - netx_plan.json : per build and stream the three arms run on the node, the host decisions of each arm (bitmap hex,
   transmissions) for the integer build, the predicted energy per round, and the CAPS and compute-blind choices (seed 0
   lightweight, the stored CNN-LSTM seed) with the node's own report energy."""
import os, sys, json, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "adaptA")); sys.path.insert(0, os.path.join(SC, "stage17/run")); sys.path.insert(0, os.path.join(SC, "iotlab/fixed"))
import numpy as np, oracle_switch as O, frozen_benchmark as fb, qemu as Q
H = json.load(open(os.path.join(HERE, "netx_host.json"))); R = H["R"]; LB = 12; TAU = 0.10
STREAMS = ["taxi", "exchange_cpc", "machine_temp", "cpu"]
ARMID = {"persistence": 0, "expsm": 1, "linear_extrap": 2, "lightweight": 3, "cnn_lstm": 4}
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
ACCQ = json.load(open(os.path.join(SC, "iotlab/fixed/accuracy_q.json")))
def pf(f):
    p = json.load(open(os.path.join(SC, f)))["per_forecast"]
    m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
    return {m.get(k, k): v["E_inc_J"] for k, v in p.items()}
BUILDS = {"m3_float": pf("iotlab/iotlab_m3_summary.json"), "m3_int": pf("iotlab/fixed/fixed_m3_21_accuracy_q_summary.json"),
          "nrf": pf("iotlab/iotlab_nrf_summary.json")}
EREP = {"m3_float": json.load(open(os.path.join(SC, "iotlab/radio/radio_m3_summary.json")))["variants"]["S_UC"]["E_inc_J"],
        "nrf": json.load(open(os.path.join(SC, "iotlab/radio/radio_nrf_summary.json")))["variants"]["S_UC"]["E_inc_J"]}
EREP["m3_int"] = EREP["m3_float"]

def bits_hex(d):
    b = bytearray((len(d) + 7) // 8)
    for r, v in enumerate(d):
        if v: b[r >> 3] |= 1 << (r & 7)
    return b.hex()

# ---- float table
T = ["#include \"netx_models.h\"\n",
     "typedef struct { const float *X, *MX, *ISX, *EW; float MY, SY, UNIT; int K, ref;\n"
     "  const float *L_K1, *L_B1, *L_K2, *L_B2, *C_CK, *C_CB, *C_LK, *C_LRK, *C_LB, *C_DK, *C_DB; } netx_stream_t;\n",
     "static const char *const NETX_NAME[NETX_NS] = {%s};\n" % ",".join('"%s"' % s for s in STREAMS),
     "static const netx_stream_t NS[NETX_NS] = {\n"]
for i, ds in enumerate(STREAMS):
    ref = H["streams"][ds]["ref"]; assert len(set(ref.values())) == 1, ref   # same reference on the three builds
    T.append("  { X%d, MX%d, ISX%d, EW%d, MY%d, SY%d, UNIT%d, K%d, %d, L_K1_%d, L_B1_%d, L_K2_%d, L_B2_%d, C_CK_%d, C_CB_%d, C_LK_%d, C_LRK_%d, C_LB_%d, C_DK_%d, C_DB_%d },\n"
             % ((i,) * 8 + (ARMID[ref["m3_float"]],) + (i,) * 11))
T.append("};\n")
open(os.path.join(HERE, "app/netx_table.h"), "w").write("".join(T))

# ---- integer build
def arr(t, name, a):
    a = np.asarray(a).reshape(-1); return "static const %s %s[%d] = {%s};\n" % (t, name, a.size, ",".join(str(int(v)) for v in a))
HQ = ["#include <stdint.h>\n#define NETX_R %d\n#define NETX_NS %d\n" % (R, len(STREAMS))]; TQ = []; plan = {"R": R, "tau": TAU, "streams": {}}
for si, ds in enumerate(STREAMS):
    S = H["streams"][ds]
    os.chdir(O.RUN); A, unit = O.load(ds); os.chdir(HERE)
    A = np.asarray(A, float).reshape(-1); Tn = A.shape[0]; tr, va = int(fb.SPLIT[0] * Tn), int(fb.SPLIT[1] * Tn)
    X = A[va - LB: va - LB + LB + R]
    Xtr, Ytr = fb.windows(A[:, None], A[:, None], LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    fi = Q.in_frac(np.nanmax(np.abs(A))); norm = Q.Norm(mx, sx, my, sy, fi)
    wl = np.load(os.path.join(SC, "adaptA/weights", f"{ds}_lightweight_s0.npz")); L = [wl["arr_%d" % i] for i in range(4)]
    wc = np.load(os.path.join(SC, "stage17/run/weights", f"{ds}_cnn_lstm_s{S['cnn_seed']}.npz")); C = [wc["arr_%d" % i] for i in range(7)]
    bt = MS[ds]["by_tau"]["0.1"]
    lq, cq = Q.LightQ(L, norm), Q.CnnQ(C, norm)
    ew = fb.make_expsm(bt["alpha"])(np.eye(LB)); eq = Q.DotQ(ew, fi); lk = Q.DotQ(Q.lin_coef(bt["K"]), fi)
    XQ = Q.q(X, fi); thr = int(math.floor(TAU * unit * 2.0 ** fi))
    F = {"persistence": lambda w: w[:, -1], "expsm": eq, "linear_extrap": lk, "lightweight": lq, "cnn_lstm": cq}
    ref = S["ref"]["m3_int"]; arms = [ref, "lightweight", "cnn_lstm"]
    st = {"fi": fi, "thr_q": thr, "arms_int": {}}
    for a in arms:
        Rq = XQ.copy(); d = []; mg = []
        for t in range(LB, LB + R):
            p = int(F[a](Rq[None, t - LB:t])[0]); e = abs(int(XQ[t]) - p)
            tx = e > thr; d.append(1 if tx else 0); mg.append(abs(e - thr) / 2.0 ** fi / unit); Rq[t] = XQ[t] if tx else p
        st["arms_int"][a] = {"tx": sum(d), "bits": bits_hex(d), "min_margin": min(mg)}
    plan["streams"][ds] = st
    for nm, a, lim in (("W1", lq.W1, 32767), ("W2", lq.W2, 32767), ("CK", cq.CK, 32767), ("LK", cq.LK, 32767), ("LRK", cq.LRK, 32767),
                       ("DK", cq.DK, 32767), ("B1", lq.b1, 2**31 - 1), ("CB", cq.CB, 2**31 - 1), ("LB", cq.LB, 2**31 - 1), ("XQ", XQ, 2**31 - 1)):
        assert np.abs(a).max() <= lim, (ds, nm)
    HQ.append(arr("int32_t", f"XQ{si}", XQ) + arr("int64_t", f"MXQ{si}", norm.MX) + arr("int64_t", f"ISXQ{si}", norm.ISX))
    HQ.append(arr("int64_t", f"EQ{si}", eq.C) + arr("int64_t", f"LQ{si}", lk.C))
    HQ.append(arr("int16_t", f"W1Q{si}", lq.W1) + arr("int32_t", f"B1Q{si}", lq.b1) + arr("int16_t", f"W2Q{si}", lq.W2))
    HQ.append(arr("int16_t", f"CKQ{si}", cq.CK) + arr("int32_t", f"CBQ{si}", cq.CB) + arr("int16_t", f"LKQ{si}", cq.LK) + arr("int16_t", f"LRKQ{si}", cq.LRK)
              + arr("int32_t", f"LBQ{si}", cq.LB) + arr("int16_t", f"DKQ{si}", cq.DK))
    TQ.append("  { XQ%d, MXQ%d, ISXQ%d, EQ%d, LQ%d, %dLL, %dLL, %d, %d, %d, %d, W1Q%d, B1Q%d, W2Q%d, %d, %d, %d, CKQ%d, CBQ%d, LKQ%d, LRKQ%d, LBQ%d, DKQ%d, %d, %d, %d, %d },\n"
              % (si, si, si, si, si, int(norm.SY), int(norm.MY), fi, norm.si, bt["K"], thr, si, si, si, int(lq.b2), lq.s1, lq.s2,
                 si, si, si, si, si, si, int(cq.DB), cq.sc, cq.sl, cq.sd) )
    # ---- CAPS and compute-blind choice per build, with the node's own report energy (fixed before any run)
    Ld = json.load(open(os.path.join(SC, "stage17/run", f"learned_{ds}.json")))["entries"]
    v_l = next(e["val_rho"] for e in Ld if e["predictor"] == "lightweight" and e["seed"] == 0)
    v_c = next(e["val_rho"] for e in Ld if e["predictor"] == "cnn_lstm" and e["seed"] == S["cnn_seed"])
    st["choice"] = {}; st["pred"] = {}
    for b, E in BUILDS.items():
        ER = EREP[b]; vl = ACCQ[ds]["seeds"]["0"]["val_q"] if b == "m3_int" else v_l
        h3 = ["persistence", "expsm", "linear_extrap"]
        ea = {"persistence": E["persistence"], "expsm": E["expsm"], "linear_extrap": E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])}
        rN = S["ref"][b]                          # the arm run on the node (paper rule at er = 3.8e-4, gen.py)
        rC = min(h3, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], h3.index(a)))   # CAPS reference with the node's report energy
        vR, eR = bt["val"][rC], ea[rC]
        El = {"lightweight": E["lightweight"], "cnn_lstm": E["cnn_lstm"]}; vv = {"lightweight": vl, "cnn_lstm": v_c}
        opts = [(rC, 0.0)] + [(k, (vv[k] - vR) * ER - (El[k] - eR)) for k in El if not (El[k] - eR >= (1 - vR) * ER)]
        caps = max(opts, key=lambda o: o[1])[0]
        bR = bt["selected"]
        blind = max([(bR, bt["val"][bR], 0)] + [(k, vv[k], {"lightweight": 208, "cnn_lstm": 99488}[k]) for k in El], key=lambda o: (o[1], -o[2]))[0]
        tx = {a: (st["arms_int"][a]["tx"] if b == "m3_int" else S["arms"][a]["tx"]) for a in [rN, "lightweight", "cnn_lstm"]}
        st["pred"][b] = {a: (ea[a] if a in ea else E[a]) + tx[a] / R * ER for a in tx}
        st["choice"][b] = {"caps": caps, "caps_ref": rC, "blind": blind, "node_ref": rN,
                           "caps_on_node": caps in tx, "blind_on_node": blind in tx}
open(os.path.join(HERE, "app/netx_models_q.h"), "w").write("".join(HQ) + (
    "typedef struct { const int32_t *XQ; const int64_t *MXQ, *ISXQ, *EQ, *LQ; int64_t SYQ, MYQ; int FI, SI, K; int32_t THR;\n"
    "  const int16_t *W1Q; const int32_t *B1Q; const int16_t *W2Q; int32_t B2Q; int S1, S2;\n"
    "  const int16_t *CKQ; const int32_t *CBQ; const int16_t *LKQ, *LRKQ; const int32_t *LBQ; const int16_t *DKQ; int32_t DBQ; int SC, SL, SD; } netx_q_t;\n"
    "static const netx_q_t NQ[%d] = {\n" % len(STREAMS)) + "".join(TQ) + "};\n" + arr("int16_t", "SIGT", Q.SIG) + arr("int16_t", "TANHT", Q.TANH))
json.dump(plan, open(os.path.join(HERE, "netx_plan.json"), "w"), indent=1)
for ds, st in plan["streams"].items():
    print(ds, "fi", st["fi"], {a: (v["tx"], round(v["min_margin"], 6)) for a, v in st["arms_int"].items()})
    for b in BUILDS: print("   ", b, st["choice"][b], {a: round(e * 1e6, 1) for a, e in st["pred"][b].items()})
