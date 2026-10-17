"""Bit-exact Python emulator of the integer predictors in main_q.c (PREREG13).
All arithmetic in int64 numpy; right shifts are arithmetic (floor), as gcc on ARM."""
import math
import numpy as np

LB = 12; ACT = 11                      # activations in Q11
LUT_N, LUT_LO, LUT_SH = 512, -8 * 2048, 6   # Q11 range [-8, 8) has 32768 codes; >> 6 gives 512 entries (step 1/32)

def q(x, s): return np.round(np.asarray(x, float) * 2.0 ** s).astype(np.int64)
def wshift(maxabs, bits=12): return int(math.floor(math.log2((2 ** bits - 1) / max(maxabs, 1e-30))))
def rshift(v, s): return v >> s if s >= 0 else v * (1 << (-s))
def in_frac(maxabs):                   # input fraction bits so that |x| * 2^f fits int32, at most 16
    return int(min(16, math.floor(math.log2((2 ** 31 - 1) / max(maxabs, 1e-30)))))

def make_lut(fn):
    centers = (LUT_LO + (np.arange(LUT_N) << LUT_SH) + (1 << (LUT_SH - 1))) / 2048.0
    return np.round(fn(centers) * 2048).astype(np.int64)
SIG = make_lut(lambda x: 1 / (1 + np.exp(-x))); TANH = make_lut(np.tanh)
def lut(T, x):
    i = (np.clip(x, LUT_LO, -LUT_LO - 1) - LUT_LO) >> LUT_SH
    return T[i]

class Norm:
    def __init__(self, mx, sx, my, sy, fi):
        self.fi = fi; self.si = 20
        self.MX = q(mx, fi); self.ISX = q(1.0 / np.asarray(sx), self.si)
        self.SY = q(sy, fi); self.MY = q(my, fi)
    def z(self, xq):                   # Q(fi) int -> Q11 saturated to int16
        z = rshift((xq - self.MX) * self.ISX, self.fi + self.si - ACT)
        return np.clip(z, -32768, 32767)
    def out(self, acc, scale):         # accumulator in Q(scale) of normalised output -> Q(fi) reading
        o16 = rshift(acc, scale - 16)
        return rshift(o16 * self.SY, 16) + self.MY

class LightQ:
    def __init__(self, P, norm):
        W1, b1, W2, b2 = [np.asarray(a, float) for a in P]
        self.n = norm; self.s1 = wshift(np.abs(W1).max()); self.s2 = wshift(np.abs(W2).max())
        self.W1 = q(W1, self.s1); self.b1 = q(b1, ACT + self.s1)
        self.W2 = q(W2[:, 0], self.s2); self.b2 = q(b2[0], ACT + self.s2)
    def __call__(self, xq):            # xq: (n, 12) int64 in Q(fi)
        z = self.n.z(xq)
        a1 = self.b1 + z @ self.W1
        h = rshift(np.maximum(a1, 0), self.s1)
        a2 = self.b2 + h @ self.W2
        return self.n.out(a2, ACT + self.s2)

class CnnQ:
    def __init__(self, P, norm):
        CK, CB, LK, LRK, LBias, DK, DB = [np.asarray(a, float) for a in P]
        self.n = norm
        self.sc = wshift(np.abs(CK).max()); self.CK = q(CK[:, 0, :], self.sc); self.CB = q(CB, ACT + self.sc)
        self.sl = wshift(max(np.abs(LK).max(), np.abs(LRK).max()))
        self.LK = q(LK, self.sl); self.LRK = q(LRK, self.sl); self.LB = q(LBias, ACT + self.sl)
        self.sd = wshift(np.abs(DK).max()); self.DK = q(DK[:, 0], self.sd); self.DB = q(DB[0], ACT + self.sd)
    def __call__(self, xq):
        n = xq.shape[0]; z = self.n.z(xq)
        zp = np.concatenate([np.zeros((n, 1), np.int64), z, np.zeros((n, 1), np.int64)], 1)
        H = np.empty((n, LB, 32), np.int64)
        for t in range(LB):
            a = self.CB + sum(zp[:, t + k, None] * self.CK[k] for k in range(3))
            H[:, t] = rshift(np.maximum(a, 0), self.sc)
        h = np.zeros((n, 32), np.int64); c = np.zeros((n, 32), np.int64)
        for t in range(LB):
            g = rshift(self.LB + H[:, t] @ self.LK + h @ self.LRK, self.sl)
            ig, fg, cc, og = lut(SIG, g[:, :32]), lut(SIG, g[:, 32:64]), lut(TANH, g[:, 64:96]), lut(SIG, g[:, 96:])
            c = (fg * c + ig * cc) >> ACT
            h = (og * lut(TANH, c)) >> ACT
        return self.n.out(self.DB + h @ self.DK, ACT + self.sd)

class DotQ:                            # expsm and linear extrapolation as fixed coefficient dot products
    def __init__(self, coef, fi):
        self.fi = fi; self.C = q(coef, 16)
    def __call__(self, xq):
        return rshift(xq @ self.C, 16)

def lin_coef(k):
    t = np.arange(k); tc = t - t.mean(); den = (tc ** 2).sum()
    # forecast = sl*k + mean - sl*tmean, with sl = sum(W*tc)/den  ->  linear in W
    c = tc / den * (k - t.mean()) + 1.0 / k
    out = np.zeros(LB); out[LB - k:] = c; return out

def expsm_coef(alpha):
    w = alpha * (1 - alpha) ** np.arange(LB); return w[::-1] / w.sum()

def forecaster(model, fi):
    """Wrap an integer model as a float forecaster for dual_prediction_suppress: (N, 12) float -> (N,) float."""
    def f(W):
        xq = q(W, fi)
        return model(xq).astype(float) / 2.0 ** fi
    return f
