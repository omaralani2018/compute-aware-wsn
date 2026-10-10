"""Generate params.h for one predictor and compute the Python reference outputs.
usage: gen.py KIND [npz] -> writes params.h, ref.txt"""
import sys, numpy as np
sys.path.insert(0, "..")
import fastfwd
KIND = sys.argv[1]
A = np.load("../intel_time.npy")[:, 0].astype(np.float64)
T = len(A); tr, va = int(0.5 * T), int(0.7 * T)
rng = np.random.default_rng(0)
starts = rng.choice(np.arange(va, T - 12), 16, replace=False)
WIN = np.stack([A[s:s + 12] for s in starts]).astype(np.float32)
TW = np.stack([A[s:s + 12] for s in range(0, tr - 12, 7)])
MX = TW.mean(0).astype(np.float32); SX = (TW.std(0) + 1e-8).astype(np.float32); ISX = (1.0 / SX).astype(np.float32)
MY, SY = np.float32(A[:tr].mean()), np.float32(A[:tr].std())
def arr(name, a, fmt="%.9ef"):
    a = np.asarray(a, np.float32)
    if a.ndim == 1: return f"static const float {name}[{a.shape[0]}] = {{" + ",".join(fmt % v for v in a) + "};\n"
    rows = ["{" + ",".join(fmt % v for v in r) + "}" for r in a]
    return f"static const float {name}[{a.shape[0]}][{a.shape[1]}] = {{" + ",".join(rows) + "};\n"
h = []
code = {"persistence": 0, "lightweight": 1, "cnn_lstm": 2, "expsm": 3, "linear_extrap": 4}[KIND]
h.append(f"#define KIND {code}\n#define NWIN {len(WIN)}\n")
h.append(arr("WIN", WIN))
z = (WIN - MX) * ISX                                      # float32, as in C
if KIND in ("lightweight", "cnn_lstm"):
    W = np.load(sys.argv[2]); W = [W[f"arr_{i}"] for i in range(len(W.files))]
    h.append(arr("MX", MX) + arr("ISX", ISX) + f"#define MY {MY:.9e}f\n#define SY {SY:.9e}f\n")
    if KIND == "lightweight":
        k1, b1, k2, b2 = W
        h.append(arr("K1", k1) + arr("B1", b1) + arr("K2", k2[:, 0]) + f"#define B2 {b2[0]:.9e}f\n")
        P = ("lightweight", k1, b1, k2, b2)
    else:
        ck, cb, lk, lrk, lb, dk, db = W
        h.append(arr("CK", ck[:, 0, :]) + arr("CB", cb) + arr("LK", lk) + arr("LRK", lrk) + arr("LB", lb)
                 + arr("DK", dk[:, 0]) + f"#define DB {db[0]:.9e}f\n")
        P = ("cnn_lstm", ck, cb, lk, lrk, lb, dk, db)
    ref = fastfwd.forward(tuple(x if isinstance(x, str) else x.astype(np.float64) for x in P), z.astype(np.float64)) * float(SY) + float(MY)
elif KIND == "expsm":
    a = 0.3; w = a * (1 - a) ** np.arange(12); w = (w[::-1] / w.sum()).astype(np.float32)
    h.append(arr("EW", w)); ref = (WIN.astype(np.float64) * w).sum(1)
elif KIND == "linear_extrap":
    k = 2; t = np.arange(k); tc = t - t.mean(); den = (tc ** 2).sum()
    h.append(f"#define LINK {k}\n#define INVK {1.0/k:.9e}f\n#define INVDEN {1.0/den:.9e}f\n#define TMEAN {t.mean():.9e}f\n" + arr("TC", tc))
    Wk = WIN[:, -k:].astype(np.float64); sl = ((Wk - Wk.mean(1, keepdims=True)) * tc).sum(1) / den
    ref = sl * k + (Wk.mean(1) - sl * t.mean())
else:
    ref = WIN[:, -1].astype(np.float64)
open("params.h", "w").write("".join(h))
np.savetxt("ref.txt", ref, fmt="%.9e")
