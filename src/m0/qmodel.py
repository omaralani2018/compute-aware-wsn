"""Fixed-point parameters for model_q15.c and a bit-exact NumPy emulator of it.
usage: qmodel.py KIND [npz]  -> writes qparams.h, qref.txt (emulated outputs), fref.txt (float outputs)"""
import sys, numpy as np
sys.path.insert(0, "..")
import fastfwd
QA, QW = 12, 12                                   # activations Q12, weights Q12
def q(a, s): return np.round(np.asarray(a, np.float64) * (1 << s)).astype(np.int64)
xs = (np.arange(513) * 128 - 32768) / 4096.0       # 513 knots over [-8, 8]
SIG = q(1 / (1 + np.exp(-xs)), QA); TNH = q(np.tanh(xs), QA)
def sat16(v): return np.clip(v, -32768, 32767)
def lut(T, x):
    x = np.clip(x, -32768, 32767); u = x + 32768; i = u >> 7; fr = u & 127
    return T[i] + (((T[i + 1] - T[i]) * fr) >> 7)
class Emu:
    """Bit-exact model of model_q15.c. Readings enter as integers x * 2^px; the
    standardisation and its inverse use per-stream integer multipliers and shifts
    chosen so that no int32 product can overflow for deviations up to 4x the
    largest training deviation."""
    def __init__(s, P, MX, ISX, MY, SY, dev):
        s.P = P; s.maxacc = 0
        s.px = int(np.floor(np.log2(2**15 / (4.0 * dev))))
        s.MXQ = q(MX, s.px) if s.px >= 0 else np.round(np.asarray(MX) / 2.0**-s.px).astype(np.int64)
        K = np.asarray(ISX, float) * 2.0 ** (QA - s.px)
        s.SH = int(np.floor(np.log2(32767 / K.max()))); assert s.SH >= 0, s.SH
        s.ISXQ = np.round(K * 2.0 ** s.SH).astype(np.int64)
        s.MYQ = int(np.round(MY * 2.0 ** s.px))
        k2 = SY * 2.0 ** (s.px - QA); s.SH2 = int(np.floor(np.log2(32767 / k2))); assert s.SH2 >= 0, s.SH2
        s.SYQ = int(np.round(k2 * 2.0 ** s.SH2))
    def toint(s, X): return np.round(np.asarray(X, float) * 2.0 ** s.px).astype(np.int64)
    def fromint(s, Y): return np.asarray(Y, float) / 2.0 ** s.px
    def acc(s, a): s.maxacc = max(s.maxacc, int(np.abs(a).max())); return a
    def __call__(s, XQ):                           # XQ: (B, 12) int64 readings, x * 2^px (per-stream scale)
        P = s.P; z = sat16(s.acc((XQ - s.MXQ) * s.ISXQ) >> s.SH)
        if P[0] == "lightweight":
            _, K1, B1, K2, B2 = P
            a = s.acc(B1[None, :] + z @ K1) >> QW; h = np.where(a > 0, sat16(a), 0)
            y = s.acc(B2 + h @ K2) >> QW
            return s.MYQ + (s.acc(y * s.SYQ) >> s.SH2)
        _, CK, CB, LK, LRK, LB, DK, DB = P
        B = XQ.shape[0]; zp = np.zeros((B, 14), np.int64); zp[:, 1:13] = z
        H = np.zeros((B, 12, 32), np.int64)
        for t in range(12):
            a = s.acc(CB[None, :] + zp[:, t:t + 3] @ CK) >> QW; H[:, t] = np.where(a > 0, sat16(a), 0)
        h = np.zeros((B, 32), np.int64); c = np.zeros((B, 32), np.int64)
        for t in range(12):
            g = s.acc(LB[None, :] + H[:, t] @ LK + h @ LRK) >> QW
            ig, fg, cc, og = lut(SIG, g[:, :32]), lut(SIG, g[:, 32:64]), lut(TNH, g[:, 64:96]), lut(SIG, g[:, 96:])
            c = sat16(s.acc(fg * c + ig * cc) >> QA)
            h = sat16(s.acc(og * lut(TNH, c)) >> QA)
        y = s.acc(DB + h @ DK) >> QW
        return s.MYQ + (s.acc(y * s.SYQ) >> s.SH2)
def quantise(kind, W):
    if kind == "lightweight":
        k1, b1, k2, b2 = W
        return ("lightweight", q(k1, QW), q(b1, QA + QW), q(k2[:, 0], QW), int(q(b2[0], QA + QW)))
    ck, cb, lk, lrk, lb, dk, db = W
    return ("cnn_lstm", q(ck[:, 0, :], QW), q(cb, QA + QW), q(lk, QW), q(lrk, QW), q(lb, QA + QW), q(dk[:, 0], QW), int(q(db[0], QA + QW)))
if __name__ == "__main__":
    kind = sys.argv[1]
    import frozen_benchmark as fb
    AA = np.load("../intel_time.npy").astype(np.float64)
    A = AA[:, 0]; T = len(A); tr, va = int(0.5 * T), int(0.7 * T)
    rng = np.random.default_rng(0); starts = rng.choice(np.arange(va, T - 12), 16, replace=False)
    WIN = np.stack([A[s:s + 12] for s in starts])
    # deployed standardisation constants (frozen_benchmark.run_dataset)
    Xtr, Ytr = fb.windows(AA, AA, 12, tr, np.random.default_rng(0))
    MX = Xtr.mean(0); SX = Xtr.std(0) + 1e-8; ISX = 1 / SX; MY, SY = float(Ytr.mean()), float(Ytr.std() + 1e-8)
    DEV = max(np.abs(Xtr - MX).max(), np.abs(Ytr - MY).max())
    PX = int(np.floor(np.log2(2**15 / (4.0 * DEV))))
    WINQ = np.round(WIN * 2.0 ** PX).astype(np.int64)
    def carr(name, a, t="int16_t"):
        a = np.asarray(a)
        if a.ndim == 1: return f"static const {t} {name}[{len(a)}] = {{" + ",".join(str(int(v)) for v in a) + "};\n"
        return f"static const {t} {name}[{a.shape[0]}][{a.shape[1]}] = {{" + ",".join("{" + ",".join(str(int(v)) for v in r) + "}" for r in a) + "};\n"
    code = {"persistence": 0, "lightweight": 1, "cnn_lstm": 2, "expsm": 3, "linear_extrap": 4}[kind]
    h = [f"#define KIND {code}\n#define NWIN {len(WIN)}\n", carr("WINQ", WINQ, "int32_t")]
    if kind in ("lightweight", "cnn_lstm"):
        Wn = np.load(sys.argv[2]); W = [Wn[f"arr_{i}"] for i in range(len(Wn.files))]
        P = quantise(kind, W); emu = Emu(P, MX, ISX, MY, SY, DEV); assert emu.px == PX
        h.append(carr("MXQ", emu.MXQ, "int32_t") + carr("ISXQ", emu.ISXQ, "int32_t")
                 + f"#define MYQ {emu.MYQ}\n#define SYQ {emu.SYQ}\n#define SH {emu.SH}\n#define SH2 {emu.SH2}\n")
        assert np.abs(emu.ISXQ).max() < 2**15 and emu.SYQ < 2**15
        if kind == "lightweight":
            _, K1, B1, K2, B2 = P
            h.append(carr("K1Q", K1) + carr("B1Q", B1, "int32_t") + carr("K2Q", K2) + f"#define B2Q {B2}\n")
        else:
            _, CK, CB, LK, LRK, LB, DK, DB = P
            h.append(carr("CKQ", CK) + carr("CBQ", CB, "int32_t") + carr("LKQ", LK) + carr("LRKQ", LRK) + carr("LBQ", LB, "int32_t")
                     + carr("DKQ", DK) + f"#define DBQ {DB}\n" + carr("SIG", SIG) + carr("TNH", TNH))
        yq = emu(WINQ); f = fastfwd.forward((kind,) + tuple(W), (WIN - MX) * ISX) * SY + MY
        print("px", PX, "SH", emu.SH, "SH2", emu.SH2, "max |acc| =", emu.maxacc, "(int32 limit 2147483647)")
    elif kind == "expsm":
        a = 0.3; w = a * (1 - a) ** np.arange(12); w = w[::-1] / w.sum(); EWQ = q(w, 12)
        h.append(carr("EWQ", EWQ, "int32_t")); yq = (WINQ * EWQ).sum(1) >> 12; f = (WIN * w).sum(1)
    elif kind == "linear_extrap":
        yq = WINQ[:, 11] + (WINQ[:, 11] - WINQ[:, 10]); f = 2 * WIN[:, 11] - WIN[:, 10]
    else:
        yq = WINQ[:, 11]; f = WIN[:, 11]
    open("qparams.h", "w").write("".join(h))
    np.savetxt("qref.txt", yq, fmt="%d"); np.savetxt("fref.txt", f, fmt="%.9e")
    print("fixed vs float, max |error| in physical units:", float(np.abs(yq / 2.0 ** PX - f).max()))
