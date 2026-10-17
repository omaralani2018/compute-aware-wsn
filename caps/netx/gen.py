"""NETX generator: per-stream data (612 readings from the start of the test span), weights, normalisation, the
reference arm per build, host closed-loop decisions (float64 and float32) and their CRC32, predicted energies."""
import os, sys, json, zlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "adaptA")); sys.path.insert(0, os.path.join(SC, "stage17/run"))
import numpy as np, oracle_switch as O, frozen_benchmark as fb, fastfwd
LB, TAU, R = 12, 0.10, 480
STREAMS = ["taxi", "exchange_cpc", "machine_temp", "cpu"]
CNN_SEED = {"taxi": 5, "exchange_cpc": 19, "machine_temp": 0, "cpu": 0}
MS = json.load(open(os.path.join(SC, "stage17/run/ms_numbers.json")))["datasets"]
def pf(f):
    p = json.load(open(os.path.join(SC, f)))["per_forecast"]; key = "E_inc_J" if "E_inc_J" in next(iter(p.values())) else None
    m = {"expsm_q": "expsm", "lin2_q": "lin2", "lin12_q": "lin12", "light_q": "lightweight", "cnn_q": "cnn_lstm"}
    return {m.get(k, k): (v[key] if key else v["E_uJ"] * 1e-6) for k, v in p.items()}
BUILDS = {"m3_float": pf("iotlab/iotlab_m3_summary.json"), "m3_int": pf("iotlab/fixed/fixed_m3_21_accuracy_q_summary.json"),
          "nrf": pf("iotlab/iotlab_nrf_summary.json")}
EREP = {"m3_float": json.load(open(os.path.join(SC, "iotlab/radio/radio_m3_summary.json")))["variants"]["S_UC"]["E_inc_J"],
        "nrf": json.load(open(os.path.join(SC, "iotlab/radio/radio_nrf_summary.json")))["variants"]["S_UC"]["E_inc_J"]}
EREP["m3_int"] = EREP["m3_float"]; ER = 3.8e-4
def ref_for(bt, E):
    arms = ["persistence", "expsm", "linear_extrap"]
    ea = {"persistence": E["persistence"], "expsm": E["expsm"], "linear_extrap": E["lin2"] + (bt["K"] - 2) / 10 * (E["lin12"] - E["lin2"])}
    return min(arms, key=lambda a: ((1 - bt["val"][a]) * ER + ea[a], arms.index(a))), ea
def run_loop(X, fore, unit, dtype):
    Rm = X.astype(dtype).copy(); dec = []
    for t in range(LB, LB + R):
        p = dtype(fore(Rm[t - LB:t]))
        tx = not (abs(dtype(X[t]) - p) / dtype(unit) <= dtype(TAU))
        dec.append(1 if tx else 0); Rm[t] = X[t] if tx else p
    return dec, Rm
def margin(X, fore, unit):
    Rm = X.astype(float).copy(); m = []
    for t in range(LB, LB + R):
        p = float(fore(Rm[t - LB:t])); d = abs(X[t] - p) / unit; m.append(abs(d - TAU))
        Rm[t] = X[t] if d > TAU else p
    return min(m)
out = {"R": R, "streams": {}}; H = ["#define NETX_R %d\n#define NETX_LB %d\n#define NETX_NS %d\n" % (R, LB, len(STREAMS))]
def arr(name, a): a = np.asarray(a, np.float32).reshape(-1); return "static const float %s[%d] = {%s};\n" % (name, a.size, ",".join("%.9g" % v for v in a))
for si, ds in enumerate(STREAMS):
    os.chdir(O.RUN); A, unit = O.load(ds); os.chdir(HERE)
    A = np.asarray(A, float).reshape(-1); T = A.shape[0]; tr, va = int(fb.SPLIT[0] * T), int(fb.SPLIT[1] * T)
    X = A[va - LB: va - LB + LB + R]
    Xtr, Ytr = fb.windows(A[:, None], A[:, None], LB, tr, np.random.default_rng(0))
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    wl = np.load(os.path.join(SC, "adaptA/weights", f"{ds}_lightweight_s0.npz")); L = [wl["arr_%d" % i] for i in range(4)]
    wc = np.load(os.path.join(SC, "stage17/run/weights", f"{ds}_cnn_lstm_s{CNN_SEED[ds]}.npz")); C = [wc["arr_%d" % i] for i in range(7)]
    bt = MS[ds]["by_tau"]["0.1"]
    nn = lambda P: (lambda W: float(fastfwd.forward(P, ((np.asarray(W, float) - mx) / sx)[None, :])[0] * sy + my))
    ew = fb.make_expsm(bt["alpha"]); ar = fb.make_ar(bt["K"])
    F = {"persistence": lambda W: float(W[-1]), "expsm": lambda W: float(ew(np.asarray(W, float)[None, :])[0]),
         "linear_extrap": lambda W: float(ar(np.asarray(W, float)[None, :])[0]),
         "lightweight": nn(("lightweight", *L)), "cnn_lstm": nn(("cnn_lstm", *C))}
    S = {"unit": float(unit), "selected_1nJ": bt["selected"], "alpha": bt["alpha"], "K": bt["K"], "cnn_seed": CNN_SEED[ds], "arms": {}}
    for k, f in F.items():
        d64, _ = run_loop(X, f, unit, np.float64); d32, _ = run_loop(X, f, unit, np.float32)
        S["arms"][k] = {"tx": int(sum(d64)), "rho": 1 - sum(d64) / R, "agree32": int(sum(a == b for a, b in zip(d64, d32))),
                        "crc32": "%08x" % zlib.crc32(bytes(d64)), "min_margin": margin(X, f, unit)}
    S["ref"] = {}; S["pred"] = {}
    for b, E in BUILDS.items():
        sel, ea = ref_for(bt, E); S["ref"][b] = sel
        S["pred"][b] = {k: (ea[k] if k in ea else E[k]) + S["arms"][k]["tx"] / R * EREP[b] for k in [sel, "lightweight", "cnn_lstm"]}
    out["streams"][ds] = S
    H.append(arr(f"X{si}", X) + arr(f"MX{si}", mx) + arr(f"ISX{si}", 1.0 / sx) + f"static const float MY{si} = {my:.9g}f, SY{si} = {sy:.9g}f, UNIT{si} = {unit:.9g}f;\n")
    H.append(arr(f"EW{si}", fb.make_expsm(bt['alpha'])(np.eye(LB))) + f"static const int K{si} = {bt['K']};\n")
    H.append(arr(f"L_K1_{si}", L[0]) + arr(f"L_B1_{si}", L[1]) + arr(f"L_K2_{si}", L[2]) + arr(f"L_B2_{si}", L[3]))
    H.append(arr(f"C_CK_{si}", C[0]) + arr(f"C_CB_{si}", C[1]) + arr(f"C_LK_{si}", C[2]) + arr(f"C_LRK_{si}", C[3]) + arr(f"C_LB_{si}", C[4]) + arr(f"C_DK_{si}", C[5]) + arr(f"C_DB_{si}", C[6]))
    print(ds, "unit %.4g" % unit, {k: (v["tx"], v["agree32"], round(v["min_margin"], 6)) for k, v in S["arms"].items()}, "refs", S["ref"])
open(os.path.join(HERE, "app/netx_models.h"), "w").write("".join(H))
json.dump(out, open(os.path.join(HERE, "netx_host.json"), "w"), indent=1)
