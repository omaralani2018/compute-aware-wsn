"""Write fw/models.h (weights, normalisation, 100 Intel test windows) and ref.json (NumPy forecasts)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SC, "stage17/run"))
import numpy as np, frozen_benchmark as fb, fastfwd
A = np.load(os.path.join(SC, "stage17/run/intel_time.npy")).astype(float)
T = A.shape[0]; tr, va = int(.5 * T), int(.7 * T); LB = 12
Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
rng = np.random.default_rng(42); TEST = A[va - LB:]
idx = [(int(rng.integers(LB, TEST.shape[0])), int(rng.integers(0, TEST.shape[1]))) for _ in range(100)]
W = np.array([TEST[t - LB:t, j] for t, j in idx], dtype=np.float32)
def arr(name, a): a = np.asarray(a, np.float32).reshape(-1); return "static const float %s[%d] = {%s};\n" % (name, a.size, ",".join("%.9g" % v for v in a))
w = np.load(os.path.join(SC, "stage17/run/weights/intel_time_lightweight_s0.npz")); L = [w["arr_%d" % i] for i in range(4)]
c = np.load(os.path.join(SC, "stage17/run/weights/intel_time_cnn_lstm_s0.npz")); C = [c["arr_%d" % i] for i in range(7)]
alpha = 0.5; ew = alpha * (1 - alpha) ** np.arange(LB); ew = ew[::-1] / ew.sum()
h = "#define NWIN 100\n"
h += arr("WIN", W) + arr("MX", mx) + arr("ISX", 1.0 / sx) + "static const float MY = %.9g, SY = %.9g;\n" % (my, sy)
h += arr("EW", ew)
h += arr("L_K1", L[0]) + arr("L_B1", L[1]) + arr("L_K2", L[2]) + arr("L_B2", L[3])
h += arr("C_CK", C[0]) + arr("C_CB", C[1]) + arr("C_LK", C[2]) + arr("C_LRK", C[3]) + arr("C_LB", C[4]) + arr("C_DK", C[5]) + arr("C_DB", C[6])
open(os.path.join(HERE, "app/models.h"), "w").write(h)
PL = ("lightweight", *L); PC = ("cnn_lstm", *C)
z = (W.astype(float) - mx) / sx
ref = {"persistence": W[:, -1].tolist(), "expsm": (W.astype(float) * ew).sum(1).tolist(),
       "lin2": fb.make_ar(2)(W.astype(float)).tolist(), "lin12": fb.make_ar(12)(W.astype(float)).tolist(),
       "lightweight": (fastfwd.forward(PL, z) * sy + my).tolist(), "cnn_lstm": (fastfwd.forward(PC, z) * sy + my).tolist()}
json.dump(ref, open(os.path.join(HERE, "ref.json"), "w"))
print("shapes", [x.shape for x in L], [x.shape for x in C])
