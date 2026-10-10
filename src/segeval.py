"""Closed-loop dual-prediction evaluation on independent segments (one mote, one span, no gap longer
than the short-gap limit inside). The rule per reading is that of dual_prediction_fix.py: forecast
from the reconstructed history shared with the sink, suppress if |x - forecast| / unit <= tau, and
otherwise send x. Each segment starts on its own: its first LOOK_BACK readings are sent as
initialisation reports (the warm-up of the main study), and no history crosses a segment boundary.
A segment with at most LOOK_BACK readings is all initialisation. Segments are advanced in lockstep so
that a learned predictor is called once per step for all active segments; the result is identical to
evaluating each segment on its own (checked in test_segeval.py)."""
import numpy as np

def _forecast(f, R, LB):
    """R: (n, >=LB) reconstructed histories; returns one forecast per row."""
    if f is None: return R[:, -1].copy()
    p = np.asarray(f(R[:, -LB:]), dtype=float).reshape(-1)
    assert p.shape[0] == R.shape[0]
    return np.where(np.isfinite(p), p, np.inf)

def seg_dps(segs, f, tau, unit=1.0, LB=12, record=False):
    X = [np.asarray(s, float) for s in segs]
    L = np.array([len(x) for x in X]); n_init = int(np.minimum(L, LB).sum())
    use = np.flatnonzero(L > LB)
    recon = [x.copy() for x in X]
    sent = [np.ones(len(x), bool) for x in X]        # warm-up readings count as sent
    errv = [np.zeros(len(x)) for x in X]
    n_tx = 0; n_steps = 0; s_err = 0.0; s_err2 = 0.0; m_err = 0.0; nonfin = 0
    order = use[np.argsort(-L[use], kind="stable")]  # longest first: active set is a prefix
    Ls = L[order]
    for t in range(LB, int(Ls.max()) if len(Ls) else LB):
        k = int(np.searchsorted(-Ls, -t, side="left"))   # number of segments with L > t
        act = order[:k]
        if k == 0: break
        H = np.stack([recon[j][t - LB:t] for j in act], axis=1).T   # same memory layout as the matrix evaluator (bitwise-equal sums)
        p = _forecast(f, H, LB); nonfin += int(np.isinf(p).sum())
        x = np.array([X[j][t] for j in act]); e = np.abs(x - p)
        sup = (e / unit) <= tau
        for q, j in enumerate(act):
            if sup[q]:
                recon[j][t] = p[q]; sent[j][t] = False; errv[j][t] = e[q]
        n_tx += int((~sup).sum()); n_steps += k
        es = np.where(sup, e, 0.0); s_err += es.sum(); s_err2 += (es ** 2).sum(); m_err = max(m_err, float(es.max()))
    out = dict(rho=1.0 - n_tx / n_steps if n_steps else float("nan"), n_steps=int(n_steps), n_tx=int(n_tx),
               n_init_reports=n_init, rho_incl_init=1.0 - (n_tx + n_init) / (n_steps + n_init),
               mean_err=s_err / n_steps if n_steps else float("nan"), rmse=float(np.sqrt(s_err2 / n_steps)) if n_steps else float("nan"),
               max_err=m_err, n_nonfinite_forecasts=nonfin)
    if record: out["sent"] = sent; out["err"] = errv; out["recon"] = recon
    return out

def seg_verify(segs, make_predictor, tau, unit=1.0, LB=12):
    """Independent receiver: the sender emits packets; a separate receiver with its own predictor
    instance rebuilds every segment from those packets alone. Returns bound_holds and identity."""
    fs, fr = make_predictor(), make_predictor()
    S = seg_dps(segs, fs, tau, unit, LB, record=True)
    X = [np.asarray(s, float) for s in segs]
    rec = [np.where(sent, x, np.nan) for x, sent in zip(X, S["sent"])]    # packets only (warm-up included)
    L = np.array([len(x) for x in X]); use = np.flatnonzero(L > LB)
    order = use[np.argsort(-L[use], kind="stable")]; Ls = L[order]
    for t in range(LB, int(Ls.max()) if len(Ls) else LB):
        k = int(np.searchsorted(-Ls, -t, side="left")); act = order[:k]
        if k == 0: break
        H = np.stack([rec[j][t - LB:t] for j in act], axis=1).T
        assert np.isfinite(H).all()
        p = _forecast(fr, H, LB)
        for q, j in enumerate(act):
            if not S["sent"][j][t]: rec[j][t] = p[q]
    err = max(float(np.max(np.abs(x - r))) for x, r in zip(X, rec))
    same = all(np.allclose(a, r, rtol=0, atol=1e-9) for a, r in zip(S["recon"], rec))
    return dict(bound_holds=bool(same and err <= tau * unit + 1e-9), sender_receiver_identical=bool(same),
                max_err_at_receiver=err, rho=S["rho"])
