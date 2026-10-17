"""Write app_q/models_q.h (integer weights for Intel seed 0, 100 test windows in Q(fi)) and ref_q.json (emulator outputs)."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); SC = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(SC, "stage17/run")); sys.path.insert(0, HERE)
import numpy as np, frozen_benchmark as fb, fastfwd, qemu as Q
A = np.load(os.path.join(SC, "stage17/run/intel_time.npy")).astype(float)
T = A.shape[0]; tr, va = int(.5 * T), int(.7 * T); LB = 12
Xtr, Ytr = fb.windows(A, A, LB, tr, np.random.default_rng(0))
mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8; my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
rng = np.random.default_rng(42); TEST = A[va - LB:]
idx = [(int(rng.integers(LB, TEST.shape[0])), int(rng.integers(0, TEST.shape[1]))) for _ in range(100)]
W = np.array([TEST[t - LB:t, j] for t, j in idx], dtype=np.float32).astype(float)   # same windows as avrsim
fi = Q.in_frac(np.nanmax(np.abs(A)))
norm = Q.Norm(mx, sx, my, sy, fi)
w = np.load(os.path.join(SC, "stage17/run/weights/intel_time_lightweight_s0.npz")); L = [w["arr_%d" % i] for i in range(4)]
c = np.load(os.path.join(SC, "stage17/run/weights/intel_time_cnn_lstm_s0.npz")); C = [c["arr_%d" % i] for i in range(7)]
lq, cq = Q.LightQ(L, norm), Q.CnnQ(C, norm)
eq = Q.DotQ(Q.expsm_coef(0.5), fi); l2 = Q.DotQ(Q.lin_coef(2), fi); l12 = Q.DotQ(Q.lin_coef(12), fi)
WQ = Q.q(W, fi)
def arr(t, name, a):
    a = np.asarray(a).reshape(-1); return "static const %s %s[%d] = {%s};\n" % (t, name, a.size, ",".join(str(int(v)) for v in a))
h = "#include <stdint.h>\n#define NWIN 100\n#define FI %d\n#define SI %d\n" % (fi, norm.si)
h += arr("int32_t", "WINQ", WQ) + arr("int64_t", "MXQ", norm.MX) + arr("int64_t", "ISXQ", norm.ISX)
h += "static const int64_t SYQ = %d, MYQ = %d;\n" % (int(norm.SY), int(norm.MY))
h += arr("int64_t", "EQ", eq.C) + arr("int64_t", "L2Q", l2.C) + arr("int64_t", "L12Q", l12.C)
h += "#define S1 %d\n#define S2 %d\n" % (lq.s1, lq.s2)
h += arr("int16_t", "W1Q", lq.W1) + arr("int32_t", "B1Q", lq.b1) + arr("int16_t", "W2Q", lq.W2) + "static const int32_t B2Q = %d;\n" % int(lq.b2)
h += "#define SC %d\n#define SL %d\n#define SD %d\n" % (cq.sc, cq.sl, cq.sd)
h += arr("int16_t", "CKQ", cq.CK) + arr("int32_t", "CBQ", cq.CB) + arr("int16_t", "LKQ", cq.LK) + arr("int16_t", "LRKQ", cq.LRK)
h += arr("int32_t", "LBQ", cq.LB) + arr("int16_t", "DKQ", cq.DK) + "static const int32_t DBQ = %d;\n" % int(cq.DB)
h += arr("int16_t", "SIGT", Q.SIG) + arr("int16_t", "TANHT", Q.TANH)
for nm, a, lim in (("W1Q", lq.W1, 32767), ("W2Q", lq.W2, 32767), ("CKQ", cq.CK, 32767), ("LKQ", cq.LK, 32767), ("LRKQ", cq.LRK, 32767), ("DKQ", cq.DK, 32767),
                   ("B1Q", lq.b1, 2**31 - 1), ("CBQ", cq.CB, 2**31 - 1), ("LBQ", cq.LB, 2**31 - 1), ("WINQ", WQ, 2**31 - 1)):
    assert np.abs(a).max() <= lim, nm
os.makedirs(os.path.join(HERE, "app_q"), exist_ok=True)
open(os.path.join(HERE, "app_q/models_q.h"), "w").write(h)
out = {k: [int(v) for v in m(WQ)] for k, m in (("persistence", lambda x: x[:, -1]), ("expsm_q", eq), ("lin2_q", l2), ("lin12_q", l12), ("light_q", lq), ("cnn_q", cq))}
z = (W - mx) / sx
flt = {"light_q": fastfwd.forward(("lightweight", *L), z) * sy + my, "cnn_q": fastfwd.forward(("cnn_lstm", *C), z) * sy + my,
       "expsm_q": W @ Q.expsm_coef(0.5), "lin2_q": fb.make_ar(2)(W), "lin12_q": fb.make_ar(12)(W)}
json.dump(dict(fi=fi, out=out, float_ref={k: list(map(float, v)) for k, v in flt.items()}), open(os.path.join(HERE, "ref_q.json"), "w"))
for k, v in flt.items():
    e = np.abs(np.array(out[k]) / 2.0 ** fi - v); print("%-8s max |fixed - float| = %.2e (tau 0.10)" % (k, e.max()))
print("fi", fi, "S1 S2", lq.s1, lq.s2, "SC SL SD", cq.sc, cq.sl, cq.sd)
