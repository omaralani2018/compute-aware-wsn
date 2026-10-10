"""Exact NumPy re-implementation of the two Keras forward passes.

Purpose is speed only. It must produce the SAME numbers as Keras, so
`verify_against_keras` asserts agreement on random inputs before use, and the
benchmark refuses to run if it fails. Any change to build() invalidates this.
"""
import numpy as np

def _sig(x): return 1.0 / (1.0 + np.exp(-x))

def extract(m, kind):
    W = [w for w in m.get_weights()]
    if kind == "lightweight":
        (k1, b1, k2, b2) = W
        return ("lightweight", k1, b1, k2, b2)
    (ck, cb, lk, lrk, lb, dk, db) = W
    return ("cnn_lstm", ck, cb, lk, lrk, lb, dk, db)

def forward(P, Z):
    """Z: (B, LOOK_BACK) standardised. Returns (B,) standardised output."""
    if P[0] == "lightweight":
        _, k1, b1, k2, b2 = P
        h = np.maximum(Z @ k1 + b1, 0.0)
        return (h @ k2 + b2).reshape(-1)
    _, ck, cb, lk, lrk, lb, dk, db = P
    B, L = Z.shape
    kw, _, F = ck.shape                       # (3, 1, 32)
    # Conv1D, padding='same', stride 1, 1 input channel, ReLU
    pad = kw // 2
    Zp = np.pad(Z, ((0, 0), (pad, pad)))
    # build (B, L, kw) sliding windows then contract with the kernel
    idx = np.arange(L)[:, None] + np.arange(kw)[None, :]
    patches = Zp[:, idx]                      # (B, L, kw)
    H = np.einsum('blk,kf->blf', patches, ck[:, 0, :]) + cb
    H = np.maximum(H, 0.0)                    # (B, L, F)
    # LSTM over L steps, Keras gate order i, f, c, o
    U = lrk.shape[0]
    h = np.zeros((B, U)); c = np.zeros((B, U))
    for t in range(L):
        z = H[:, t, :] @ lk + h @ lrk + lb    # (B, 4U)
        i = _sig(z[:, :U]); f = _sig(z[:, U:2*U])
        cc = np.tanh(z[:, 2*U:3*U]); o = _sig(z[:, 3*U:])
        c = f * c + i * cc
        h = o * np.tanh(c)
    return (h @ dk + db).reshape(-1)

def verify_against_keras(m, kind, rng=None, n=64, tol=2e-4):
    rng = rng or np.random.default_rng(0)
    L = m.input_shape[1]
    Z = rng.standard_normal((n, L)).astype("float32")
    zz = Z[..., None] if kind == "cnn_lstm" else Z
    ref = np.asarray(m(zz, training=False)).reshape(-1).astype(np.float64)
    got = forward(extract(m, kind), Z.astype(np.float64))
    err = np.abs(ref - got).max()
    scale = max(1e-8, np.abs(ref).max())
    if err / scale > tol:
        raise AssertionError(
            "NumPy forward pass disagrees with Keras for %s: max abs err %.3e "
            "(relative %.3e). The fast path is NOT equivalent; do not use it."
            % (kind, err, err / scale))
    return err
