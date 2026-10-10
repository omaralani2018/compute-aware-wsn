"""
Shared-state dual-prediction suppression  --  drop-in replacement (v2, hardened).

WHY THIS EXISTS
---------------
The previous evaluation computed suppression as

    err = np.abs(f(W_test) - y_test)        # W_test built from ACTUAL readings
    rho = (err <= tau).mean()

That measures a scheme in which the node compares each reading against a forecast
made from its OWN history of readings, including readings it never transmitted.
The sink does not have those readings, cannot reproduce that forecast, and its
reconstruction error is NOT bounded by tau. On the Intel matrix at tau = 0.10 C
that scheme gives rho = 0.979 with a worst-case sink error of 0.473 C (4.7x tau).

A real dual-prediction scheme has both ends predict from the same reconstructed
series x_tilde:

    x_tilde(t) = x(t)        if a packet was sent at t
    x_tilde(t) = x_hat(t)    if round t was suppressed

State is path-dependent, so this must step through time; it cannot be batch
evaluated. It is still fast because all nodes are predicted together per step.

SCOPE AND ASSUMPTIONS (state these in the paper)
-----------------------------------------------
* Reliable, in-order delivery. No packet loss, no retransmission.
* Synchronised initialisation: the sink is assumed to hold the warm-up span
  [0, t0). Either transmit it explicitly (and charge n_init_reports below to the
  energy budget) or declare it a pre-deployment calibration handshake.
* Stateless predictors: identical input windows are sufficient to keep both ends
  in step. A stateful/recurrent predictor that carries hidden state ACROSS calls
  (e.g. Keras stateful=True) needs that state synchronised too, and this function
  does not do that. The Conv1D+LSTM used here is stateless per call, so it is fine.

TRAINING BUDGET AND PROTOCOL-AWARE TRAINING (measured, with control)
-------------------------------------------------------------------
Under shared state a learned predictor consumes its own forecasts on suppressed
rounds, while persistence re-anchors on a real transmitted reading whenever it
fires. Four arms on the Intel matrix, tau = 0.10 C, Dense(16), 5 seeds, identical
architecture and splits. A and B use the SAME checkpoint per seed:

    persistence (shared state)                          rho = 0.907

    A  actual-train(20ep) / actual-eval  [diagnostic]   rho = 0.864 +/- 0.067
    B  actual-train(20ep) / shared-state                rho = 0.653 +/- 0.099
    B+ actual-train(40ep) / shared-state                rho = 0.759 +/- 0.074
    C  protocol-aware-train(20ep) / shared-state        rho = 0.790 +/- 0.030

Paired per-seed deltas:

    B+ - B  (extra optimisation only)          = +0.106 +/- 0.068
    C  - B  (reconstructed inputs, no control) = +0.137 +/- 0.092
    C  - B+ (inputs, budget held equal)        = +0.031 +/- 0.064

What survives:
  * The evaluation change is real and large: A -> B costs 0.211. Suppression
    measured on actual-history windows is not achievable under the protocol.
  * The recovery is NOT mainly due to protocol-aware training. Once the
    optimisation budget is held equal, the reconstructed-input effect is
    +0.031 with a standard deviation of +/-0.064 -- not distinguishable from
    zero at 5 seeds. An earlier claim here that protocol-aware training
    recovered two-thirds of the loss was confounded by training budget and
    has been withdrawn.
  * Arm B is simply undertrained. EPOCHS = 20 appears insufficient for this
    model; check convergence per dataset before comparing predictors at all,
    or the whole comparison measures training budget rather than architecture.

C is one round of protocol-aware retraining, not a fixed point: retraining
changes the model that generated the reconstructed windows, so the
training/deployment mismatch is reduced, not eliminated. Report it that way.

PERFORMANCE
-----------
Call a Keras model as `m(z, training=False)`, NOT `m.predict(z)`. predict() has a
large per-call overhead and this loop calls the model once per time step.
"""

import numpy as np

__all__ = ["dual_prediction_suppress", "verify_with_independent_receiver"]


def _check_common(X, tau, look_back, train_frac, unit):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be 1-D or 2-D, got shape %r" % (X.shape,))
    if not np.isfinite(X).all():
        n = int((~np.isfinite(X)).sum())
        raise ValueError(
            "X contains %d missing/non-finite values. Reject them or implement an "
            "explicit missing-observation protocol; silently treating a NaN as a "
            "transmitted reading records zero reconstruction error and then "
            "propagates NaN into every subsequent window." % n)
    if not (np.isfinite(unit) and unit > 0):
        raise ValueError("unit must be finite and positive, got %r" % (unit,))
    if not (np.isfinite(tau) and tau >= 0):
        raise ValueError("tau must be finite and non-negative, got %r" % (tau,))
    if isinstance(look_back, bool) or not isinstance(look_back, (int, np.integer)):
        raise TypeError("look_back must be an integer, got %r (a fractional value "
                        "passes a range check and then fails at slicing)" % (look_back,))
    if not (1 <= look_back < X.shape[0]):
        raise ValueError("look_back out of range: %r" % (look_back,))
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1), got %r" % (train_frac,))
    t0 = max(look_back, int(train_frac * X.shape[0]))
    if t0 >= X.shape[0]:
        raise ValueError("no test steps: t0=%d >= T=%d" % (t0, X.shape[0]))
    return X, t0


def _forecast(predict_batch, recon, t, look_back, N, nonfinite_counter):
    if predict_batch is None:
        return recon[t - 1, :].copy()
    W = recon[t - look_back:t, :].T                      # (N, look_back), reconstructed
    pred = np.asarray(predict_batch(W), dtype=float).reshape(-1)
    if pred.shape[0] != N:
        raise ValueError(
            "predictor returned %d values for %d nodes. A length-1 return would "
            "silently broadcast across all nodes; return exactly one forecast per "
            "node." % (pred.shape[0], N))
    bad = ~np.isfinite(pred)
    if bad.any():
        # defined fallback: a forecast the sink cannot use forces a transmission
        nonfinite_counter[0] += int(bad.sum())
        pred = np.where(bad, np.inf, pred)
    return pred


def dual_prediction_suppress(X, predict_batch=None, tau=0.10, look_back=12,
                             train_frac=0.70, unit=1.0):
    """
    Suppression under a correctly synchronised dual-prediction scheme.

    predict_batch : callable or None
        (N, look_back) reconstructed history -> (N,) one-step forecasts.
        None = persistence, which under shared state is exactly send-on-delta.
        Your existing `f` from fit_predictor(...) already has this signature.
    unit : float
        Divisor applied to the error before comparing with tau. Use 1.0 for a
        physical tolerance. For a normalised tolerance pass the TRAINING-span
        std -- not the std of the whole series, which leaks test data into the
        normalisation constant.

    Returns dict: rho, mean_err, max_err, n_steps, n_tx, n_nonfinite_forecasts,
                  n_init_reports.
    """
    X, t0 = _check_common(X, tau, look_back, train_frac, unit)
    T, N = X.shape
    recon = X.copy()          # warm-up span assumed synchronised; see SCOPE above
    nonfinite = [0]
    n_tx = 0
    errs = []

    for t in range(t0, T):
        pred = _forecast(predict_batch, recon, t, look_back, N, nonfinite)
        err = np.abs(X[t, :] - pred)
        suppress = (err / unit) <= tau
        recon[t, :] = np.where(suppress, pred, X[t, :])
        errs.append(np.where(suppress, err, 0.0))
        n_tx += int((~suppress).sum())

    errs = np.concatenate(errs)
    n_steps = (T - t0) * N
    return {
        "rho": 1.0 - n_tx / n_steps,
        "mean_err": float(errs.mean()),
        "rmse": float(np.sqrt((errs ** 2).mean()),),
        "max_err": float(errs.max()),
        "n_steps": int(n_steps),
        "n_tx": int(n_tx),
        "n_nonfinite_forecasts": int(nonfinite[0]),
        "n_init_reports": int(t0 * N),   # charge these if the warm-up is transmitted
    }


def verify_with_independent_receiver(X, make_predictor, tau=0.10, look_back=12,
                                     train_frac=0.70, unit=1.0, strict=True):
    """
    Independent check. The sender decides and emits packets; a SEPARATE receiver
    reconstructs using only those packets and its own predictor instance. The
    error bound is then measured on the receiver's series, not on a quantity the
    sender computed about itself.

    make_predictor : callable -> (predict_batch or None)
        Called twice, once for the sender and once for the receiver, so the two
        sides hold independent OBJECTS wrapping IDENTICAL FROZEN WEIGHTS. It must
        NOT train a model on each call: two independently trained models have
        different weights, and this check would then fail for a reason that has
        nothing to do with the protocol. Train once, freeze, and return a fresh
        closure over that same model each time, e.g.

            m = train_once(...)                      # trained ONCE, then frozen
            make_predictor = lambda: (lambda W: predict_with(m, W))

    strict : bool
        If True (default) raise on any mismatch instead of returning flags that a
        caller might ignore.
    """
    X, t0 = _check_common(X, tau, look_back, train_frac, unit)
    T, N = X.shape
    send_fn, recv_fn = make_predictor(), make_predictor()

    sender = X.copy()
    packets = []                      # (t, mask, values) -- what actually goes on air
    nf = [0]
    for t in range(t0, T):
        pred = _forecast(send_fn, sender, t, look_back, N, nf)
        err = np.abs(X[t, :] - pred)
        sup = (err / unit) <= tau
        sender[t, :] = np.where(sup, pred, X[t, :])
        packets.append((t, ~sup, np.where(~sup, X[t, :], np.nan)))

    receiver = X.copy()               # same synchronised warm-up
    receiver[t0:, :] = np.nan
    nf2 = [0]
    for (t, sent, vals) in packets:
        pred = _forecast(recv_fn, receiver, t, look_back, N, nf2)
        receiver[t, :] = np.where(sent, vals, pred)

    same = np.allclose(sender[t0:], receiver[t0:], rtol=0, atol=1e-9)
    err_recv = np.abs(X[t0:] - receiver[t0:])
    n_sent = int(sum(int(s.sum()) for _, s, _ in packets))
    n_steps = (T - t0) * N
    if strict:
        if not same:
            raise AssertionError(
                "receiver reconstruction diverged from the sender: the two sides are "
                "not in step (check that make_predictor returns identical frozen weights)")
        if err_recv.max() > tau * unit + 1e-9:
            raise AssertionError(
                "tolerance bound violated at the receiver: max %.6f > tau*unit %.6f"
                % (err_recv.max(), tau * unit))
    return {
        "sender_receiver_identical": bool(same),
        "rho": 1.0 - n_sent / n_steps,
        "max_err_at_receiver": float(err_recv.max()),
        "bound_holds": bool(err_recv.max() <= tau * unit + 1e-9),
        "n_steps": int(n_steps),
    }


# ----------------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------------
# Intel, physical tolerance in degrees C:
#     res = dual_prediction_suppress(X, None, tau=0.10, unit=1.0)        # persistence
#     f, _ = fit_predictor('cnn_lstm', Xtr, Ytr, seed=sd)
#     res = dual_prediction_suppress(X, f,   tau=0.10, unit=1.0)         # learned
#
# NAB -- DO NOT segment first. Chopping one series into consecutive pseudo-nodes
# and then splitting 70/30 INSIDE each segment breaks chronology: segment 1's test
# window precedes segment 20's training window, so pooled training rows contain
# observations later than another segment's test period. Split the stream once,
# then segment:
#     cut  = int(TRAIN_FRAC * len(v))
#     u_tr = v[:cut].std()                       # scale from TRAINING data only
#     te   = v[cut:]
#     seg  = len(te) // N_NODES_NAB
#     Xte  = np.stack([te[i*seg:(i+1)*seg] for i in range(N_NODES_NAB)], axis=1)
#     res  = dual_prediction_suppress(Xte, f, tau=D_MAX_B, unit=u_tr, train_frac=<small warm-up>)
#
# Independent bound check (do this once per configuration, for the paper):
#     verify_with_independent_receiver(X, lambda: None, tau=0.10)
#
# Report res['mean_err'] AND res['max_err']; the max is what demonstrates the
# bound actually holds.
