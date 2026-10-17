"""
Frozen benchmark protocol -- shared-state dual-prediction, compute-aware.

STATUS (26 Sep 2026): ALL ARMS RUN. 11 datasets x {persistence, expsm, linear
extrapolation, lightweight x 20 seeds, CNN-LSTM x 20 seeds}, plus a lightweight
tolerance sweep at tau = 0.05/0.10/0.20/0.50. Every configuration passed the
independent receiver check. An independent audit recomputed the main table
(22 rows) with no discrepancies and found no test-data leakage in the code.
Energy throughout is MODELLED from MAC counts and a radio model. No hardware
measurement of inference or radio energy has been performed in this study.

DISCLOSURES REQUIRED IN THE PAPER (from the audit)
 - ALPHA_GRID and LIN_K_GRID were widened AFTER test numbers had been seen. The
   motivation was the equal-information principle (every arm tuned on validation
   by the same rule) and boundary hits observed in VALIDATION-selected alpha, but
   test scores were visible at the time, so the change is post-hoc. It favours
   the cheap arms, which is the direction of this paper's main conclusion, so it
   is NOT conservative for that conclusion: it makes "learned models lose" easier
   to obtain. (It is conservative only for the positive claims on taxi,
   exchange_cpc and traffic.) The affected comparisons are exploratory until
   confirmed on a dataset scored under a grid fixed in advance.
 - The cheap reference is the cheap arm with the best VALIDATION suppression.
   An earlier table picked it by TEST score (an oracle); that table is withdrawn.
 - Lightweight runs reproduce bit-for-bit. CNN-LSTM runs do not: a re-run of
   traffic seed 0 gave 0.0944 against 0.0885 stored, at the same epoch (thread
   count differed between runs; LSTM reductions are not order-invariant). Results
   are reproducible in distribution, not bitwise. Future runs should call
   tf.config.experimental.enable_op_determinism().
 - traffic has a 339-step test span; its +0.018 gain is about 6 transmissions.
 - Wilcoxon p-values use scipy's normal approximation where per-seed values tie.

Fix this file BEFORE looking at test results. Every knob below is a study-design
decision, recorded here so the run is reproducible and the stopping rule cannot
drift once test numbers are visible.

PROTOCOL (frozen)
-----------------
  split            chronological 50 / 20 / 30  -> train / validation / TEST
  preprocessing    scaling fitted on TRAIN only
  training cap     200 epochs  (a documented resource cap, NOT a convergence claim)
  checkpoints      every 25 epochs -> 8 checkpoints (25..200); epoch 0 not scored
  window cap       MAX_WINDOWS = 4000 sampled from the training span, and Keras
                   holds out a further 15% internally for its own loss curve.
                   That internal split is NOT used for checkpoint selection.
  selection        checkpoint with best SHARED-STATE VALIDATION suppression.
                   With architecture and energy costs fixed, maximising rho also
                   minimises modelled operating energy, so this is an energy rule.
  seeds            20, fixed, reported individually
  baselines        persistence; least-squares linear extrapolation over the
                   look-back window; exponential smoothing with alpha selected
                   on validation from ALPHA_GRID (same rule as the neural
                   checkpoints, so every arm gets equal tuning opportunity)
  learned          lightweight Dense(16), CNN-LSTM
  final            selected checkpoints evaluated ONCE on the untouched test span
  reporting        per-seed rho, MAE, RMSE, max error, non-finite forecasts,
                   and modelled energy with INITIALISATION SEPARATED from operating

TRAINING PROTOCOL OF THE LEARNED ARMS  (state this in the paper)
----------------------------------------------------------------
The neural arms here are trained on ACTUAL sensor history and then evaluated in
closed loop on the shared reconstructed state. That is the standard practice in
this literature and it is what this benchmark measures. It is NOT a claim that
actual-history training is the best available choice.

Conclusions this design supports:
    "Within a 200-epoch budget, these actual-history-trained models achieve
     rho = ... on the held-out span, at a modelled operating energy of ... ."
Conclusions it does NOT support:
    "Learned predictors are (or are not) superior to protocol-aware training."
    "No protocol-aware alternative would do better."
A four-arm study of training protocols (actual vs reconstructed history, with the
B+ equal-budget control) exists separately and is reported as a LABELLED SIDE
STUDY. C minus B+ was +0.031 +/- 0.064, reported as a MEAN AND STANDARD DEVIATION
over seeds. That is a description of spread, not a confidence interval and not a
significance test: the fact that mean minus one SD falls below zero is not by
itself evidence of no effect. The study is simply underpowered to support a
general claim either way, and it is not merged into this benchmark.

EXAMINATION HISTORY OF EACH DATASET  (read before quoting any number)
---------------------------------------------------------------------
NOTHING CURRENTLY IN THIS PIPELINE IS AN UNEXAMINED HOLDOUT. Every stream below
has been evaluated and reported before this protocol was frozen. Re-running them
under the corrected evaluator produces CORRECTED results, not independent
confirmation. That distinction is the whole point of this section.

  Ten NAB streams -- PREVIOUSLY EXAMINED.
      The submitted manuscript already reports test-span results for all ten
      (Tables 2, 7 and 8; Sections 6.2 and 6.4), under a 70/30 split, and those
      findings shaped the research direction that produced this benchmark.
      Re-scoring them here corrects a defective evaluator on data whose test
      spans have been seen. Report as: corrected results, prior exposure
      disclosed. Do NOT describe as held-out confirmation.

  Intel temperature -- PREVIOUSLY EXAMINED, twice over.
      (a) The manuscript reports Intel operating points (Table 1, Section 6.1).
      (b) During development the convergence sweep ran with a defective
          validation slice, VAL = A[va-LOOK_BACK : va+min(600, T-va)], which
          scored rows inside the test span, and its output informed design
          decisions taken afterwards. Fixing the slice does not undo that.
      Report as: corrected results, prior exposure disclosed.

  Genuinely unexamined at the time of freezing: only data never scored under any
      evaluator. Such a stream is a candidate for confirmatory evidence only if
      it is scored exactly once, after this file is frozen. Any stream
      inspected before its single scoring run joins the examined list above and
      must be relabelled here.

This does not devalue the corrected runs. A corrected measurement on examined
data is still the correct measurement, and it supersedes the defective one. It
simply is not independent confirmation, and must not be written up as though it
were.

WHAT THE COMPUTE CORRECTION DOES AND DOES NOT DETERMINE
--------------------------------------------------------
Raising the CNN-LSTM charge from 53,408 to 99,488 MACs raises Einf by ~86% and
therefore worsens its verdict AT FIXED SUPPRESSION. Suppression is not fixed:
the shared-state evaluator changes every measured rho as well, in a direction
that is not known in advance. The criterion Einf < drho*e_r has BOTH sides
moving, so the final verdict cannot be predicted from the MAC count alone and
must be measured. The same applies to the lightweight model: its MAC count of
208 is unchanged, but its measured suppression is not, so its results need
regenerating too. Nothing about either model's standing should be asserted
before these runs complete.

CLAIM DISCIPLINE
----------------
Results are "best within a 200-epoch budget", not "the best this architecture can
do". Arm A (actual-history evaluation) is a diagnostic reference only and must
never be quoted as an achievable saving. Total training cost for protocol-aware
training must include generating the reconstructed windows.

A ranking reversal was observed on Intel between the validation and test spans
(validation: linear_extrap 0.935 > persistence 0.929; test: persistence 0.844 >
linear_extrap 0.737, a ~68.4% difference in modelled operating energy). This
shows that the preferred predictor can change across periods on this stream. It
does NOT establish that validation-based selection is generally untrustworthy,
and it is reported as an observation, not as a methodological finding.
"""
import numpy as np, json, time

# ---- frozen constants -------------------------------------------------------
LOOK_BACK   = 12
TAU         = 0.10          # physical units for Intel (deg C); normalised for NAB
SPLIT       = (0.50, 0.70)  # train ends / validation ends
EPOCH_CAP   = 200
CKPT_EVERY  = 25
SEEDS       = list(range(20))   # 20, matching the reporting in the manuscript.
#   Re-frozen from 10 to 20 BEFORE any run of this protocol, so no test number
#   influenced the change. Doubling the seed budget roughly doubles runtime.
MAX_WINDOWS = 4000
E_OP        = 1e-9          # J per MAC
E_SENSE     = 1e-7          # J per round
E_R         = 3.8e-4        # J per report  (define scope in the paper; see notes)
# CNN-LSTM MACs, derived from the architecture actually built below:
#   Conv1D(32, k=3, 1 in-channel) over 12 positions      = 12*32*3*1  =  1,152
#   LSTM(32) whose INPUT IS THE CONV OUTPUT (32 channels,
#            not the raw 1-channel series) over 12 steps = 4*32*(32+32)*12 = 98,304
#   Dense(1) from 32                                     =                32
#                                                          total    =  99,488
# Crossover radio cost: e_r* = (E_model - E_ref) / drho. The numerator is the
# DIFFERENCE in inference cost, because the cheap reference also computes (12 MACs
# for smoothing, K for extrapolation). Charging the model's full count instead
# overstates e_r* slightly (by <=6% for the lightweight model, negligible for the
# CNN-LSTM). main_table.py uses the difference.
# Cross-check: parameter count (3*1*32+32) + 4*(32*32+32*32+32) + (32+1) = 8,481,
# which matches the profiled trainable-parameter count, confirming LSTM input dim 32.
# The previous figure of 53,408 came from a formula assuming LSTM input dim = 1 and
# undercounts by ~46%. Biases/activations are additional and still not charged.
MACS = {"persistence": 0, "linear_extrap": LOOK_BACK, "expsm": LOOK_BACK,
        "lightweight": 208, "cnn_lstm": 99488}

from dual_prediction_fix import (dual_prediction_suppress as dps,
                                 verify_with_independent_receiver as verify)

# Every field this module reads out of the evaluator. The guard below asserts the
# WHOLE set, so an older dual_prediction_fix.py fails at import rather than part
# way through a multi-hour run. Extend this set whenever record() reads a new key.
REQUIRED_DPS_FIELDS = {
    "rho", "mean_err", "rmse", "max_err",
    "n_steps", "n_tx", "n_nonfinite_forecasts", "n_init_reports",
}
REQUIRED_VERIFY_FIELDS = {
    "sender_receiver_identical", "rho", "max_err_at_receiver",
    "bound_holds", "n_steps",
}

def _check_evaluator():
    """Fail loudly if an OLDER dual_prediction_fix.py is on the path."""
    import numpy as _np
    # Probe trace: a monotone ramp whose PER-STEP increment (0.6*tau) is below the
    # tolerance but whose TWO-step increment (1.2*tau) is above it. A correct
    # closed-loop evaluator must therefore transmit on roughly every other step
    # (rho strictly below 1). An open-loop evaluator that compares each forecast
    # against ACTUAL history instead of the shared reconstructed state sees only
    # the small per-step increment, suppresses everything, and reports rho = 1.
    # This distinguishes the two without needing the source to be inspected.
    _TAU = 0.05
    Z = (_np.arange(60) * 0.6 * _TAU)[:, None]
    r = dps(Z, None, tau=_TAU, look_back=4, train_frac=0.5)
    missing = REQUIRED_DPS_FIELDS - set(r)
    if missing:
        raise ImportError(
            "dual_prediction_suppress on the path is an OLD version (missing %s). "
            "Use the hardened evaluator (v3) under exactly this module name."
            % sorted(missing))
    v = verify(Z, (lambda: None), tau=_TAU, look_back=4,
               train_frac=0.5, strict=True)
    missing_v = REQUIRED_VERIFY_FIELDS - set(v)
    if missing_v:
        raise ImportError(
            "verify_with_independent_receiver on the path is an OLD version "
            "(missing %s)." % sorted(missing_v))
    # behavioural checks, not just field presence: the VAL/TEST slice fix and the
    # bound itself are what an old file would silently get wrong.
    if not (0.0 <= r["rho"] <= 1.0) or r["n_steps"] <= 0:
        raise ImportError("evaluator returned an out-of-range rho / n_steps.")
    if r["max_err"] > _TAU + 1e-9:
        raise ImportError(
            "evaluator violated the tolerance bound on the probe trace "
            "(max_err %.6f > tau %.4f); it is not enforcing send-on-delta."
            % (r["max_err"], _TAU))
    if r["rho"] > 0.75:
        raise ImportError(
            "evaluator reported rho = %.3f on a ramp that must force a transmit "
            "on roughly every other step. It is almost certainly comparing "
            "forecasts against ACTUAL history rather than the shared "
            "reconstructed state (the open-loop error). Use the v3 evaluator."
            % r["rho"])
    if v["max_err_at_receiver"] > _TAU + 1e-9 or not v["bound_holds"]:
        raise ImportError(
            "independent receiver check does not enforce the bound on the probe "
            "trace; verify_with_independent_receiver is not the v3 version.")
_check_evaluator()

# PROVENANCE, NOT ENFORCEMENT.
# The correctness guarantee comes from _check_evaluator() above, which exercises
# the evaluator's behaviour. The hash below is a RECORD of which file produced a
# given run, so a result can be traced back to its code. A mismatch is REPORTED,
# not fatal: a trailing newline or a CRLF checkout changes the hash without
# changing behaviour, and a hard failure there would block legitimate runs for a
# non-reason. Read a DIFFERS line as "check what changed", not "wrong file".
EVALUATOR_SHA256 = "e8cfb5f685df10023bc3d18e63ef850da0453008f74ac5a0d5653a019dd0f4d6"

def _log_fingerprint():
    import hashlib, dual_prediction_fix as _m
    try:
        h = hashlib.sha256(open(_m.__file__, "rb").read()).hexdigest()
    except Exception as e:
        print(f"[provenance] could not hash evaluator: {e}", flush=True); return
    state = "MATCHES frozen reference" if h == EVALUATOR_SHA256 else \
            "DIFFERS from frozen reference (advisory only; behaviour checks passed)"
    print(f"[provenance] dual_prediction_fix.py sha256={h[:16]}... {state}\n"
          f"[provenance] path={_m.__file__}", flush=True)
_log_fingerprint()


# ---- deterministic predictors (no seeds, no training) -----------------------
_t = np.arange(LOOK_BACK); _tc = _t - _t.mean(); _den = (_tc ** 2).sum()

def make_ar(k=LOOK_BACK):
    t = np.arange(k); tc = t - t.mean(); den = (tc ** 2).sum()
    def f(W):
        W = np.asarray(W)[:, -k:]
        sl = ((W - W.mean(1, keepdims=True)) * tc).sum(1) / den
        return sl * k + (W.mean(1) - sl * t.mean())
    return f

def select_ar(VAL, tau, look_back, unit):
    best = (-1.0, None)
    for k in LIN_K_GRID:                       # ascending: ties keep smaller k
        r = dps(VAL, make_ar(k), tau=tau, look_back=look_back,
                train_frac=1e-6, unit=unit)["rho"]
        if r > best[0]: best = (r, k)
    return make_ar(best[1]), best[1]

# Exponential smoothing over the look-back window. alpha is SELECTED ON THE
# VALIDATION SPAN from ALPHA_GRID, by the same rule used for neural checkpoints.
#
# It was previously hard-coded at alpha = 0.9. That was a defect, not a choice:
# at 0.9 the weights are [..., 0.009, 0.09, 0.90], so 90% of the mass sits on the
# most recent sample and the predictor is barely distinguishable from
# persistence. Fixing it while tuning the neural models' checkpoints on
# validation gave the neural models a tuned hyperparameter and the cheap baseline
# none, which flattered the neural models. Measured effect of the fix, test
# suppression: rds_cpu_util 0.132 -> 0.236, cpu 0.212 -> 0.279, machine_temp
# 0.740 -> 0.770, twitter_aapl 0.687 -> 0.696. Selecting alpha costs nothing at
# inference; the charge stays at LOOK_BACK MACs.
# alpha = 1.0 is persistence EXACTLY, so it is included: when validation selects
# it, that is the honest statement "smoothing does not help on this stream",
# not a grid artifact. The earlier 11-point grid stopped at 0.99 and 4 of 11
# datasets selected a boundary value, which meant the grid was truncating.
ALPHA_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
              0.95, 0.99, 0.995, 0.999, 1.0]
# Linear extrapolation fits over the last K of the look-back window; K is
# selected on validation like alpha, so every cheap arm gets equal tuning.
# The MAC charge follows the selected K. Fixing K at 12 was costing a lot:
# on taxi, K=2 gives 0.429 test suppression against K=12's 0.115.
LIN_K_GRID = [2, 3, 4, 6, 8, 10, 12]
# TIE-BREAKS, deterministic and stated:
#   K       -> smaller K   (genuinely lower inference charge: K MACs)
#   alpha   -> smaller alpha (no compute difference; first in ascending order)
#   epoch   -> earlier epoch (no inference-cost difference -- the architecture is
#              unchanged -- purely a deterministic rule; it does save training time)

def make_expsm(alpha):
    w = alpha * (1 - alpha) ** np.arange(LOOK_BACK)
    w = (w[::-1] / w.sum())
    return lambda W: (np.asarray(W) * w).sum(1)

def select_expsm(VAL, tau, look_back, unit):
    """Pick alpha on validation suppression. Returns (predictor, alpha)."""
    best = (-1.0, None)
    for a in ALPHA_GRID:
        r = dps(VAL, make_expsm(a), tau=tau, look_back=look_back,
                train_frac=1e-6, unit=unit)["rho"]
        if r > best[0]: best = (r, a)
    return make_expsm(best[1]), best[1]


# ---- learned predictors -----------------------------------------------------
def build(kind, seed):
    from tensorflow import keras
    from tensorflow.keras import layers
    keras.utils.set_random_seed(int(seed))
    if kind == "lightweight":
        m = keras.Sequential([layers.Input((LOOK_BACK,)),
                              layers.Dense(16, activation="relu"), layers.Dense(1)])
    else:
        m = keras.Sequential([layers.Input((LOOK_BACK, 1)),
                              layers.Conv1D(32, 3, activation="relu", padding="same"),
                              layers.LSTM(32), layers.Dense(1)])
    m.compile("adam", "mse")
    return m

def wrap(m, kind, mx, sx, my, sy):
    """Closure over ONE frozen model. Call twice for sender/receiver checks."""
    def f(W):
        z = ((np.asarray(W) - mx) / sx).astype("float32")
        if kind == "cnn_lstm":
            z = z[..., None]
        return np.asarray(m(z, training=False)).reshape(-1) * sy + my
    return f


# ---- windows ----------------------------------------------------------------
def windows(S, targets, lo, hi, rng):
    Xs, Ys = [], []
    N = S.shape[1]
    for j in range(N):
        for t in range(max(LOOK_BACK, lo), hi):
            Xs.append(S[t - LOOK_BACK:t, j]); Ys.append(targets[t, j])
    Xs, Ys = np.array(Xs), np.array(Ys)
    if len(Xs) > MAX_WINDOWS:
        i = rng.choice(len(Xs), MAX_WINDOWS, replace=False); Xs, Ys = Xs[i], Ys[i]
    return Xs, Ys


# ---- energy -----------------------------------------------------------------
def energy_report(name, rho, n_steps, n_init_reports):
    e_inf = MACS[name] * E_OP
    operating = n_steps * (E_SENSE + e_inf + (1.0 - rho) * E_R)
    init = n_init_reports * E_R          # cost of synchronising the warm-up span
    return {"E_inf_per_round": e_inf, "E_operating": operating,
            "E_initialisation": init, "E_total": operating + init}


# ---- one dataset ------------------------------------------------------------
def run_dataset(A, name="dataset", unit=1.0, do_learned=True, out=None):
    """
    A : (T, N) continuous chronological stream. For a NAB series evaluate it as
        ONE stream (N=1); segmented pseudo-nodes are a separate, labelled
        experiment because the per-segment resets change the prediction task.
    """
    A = np.asarray(A, float)
    if A.ndim == 1: A = A[:, None]
    T = A.shape[0]
    tr, va = int(SPLIT[0] * T), int(SPLIT[1] * T)
    # validation is the MIDDLE span; test begins at va and is never scored here.
    VAL  = A[tr - LOOK_BACK: va]
    TEST = A[va - LOOK_BACK:]
    assert tr < va < T and VAL.shape[0] > LOOK_BACK
    rng = np.random.default_rng(0)
    res = {"dataset": name, "unit": unit, "T": T, "split": [tr, va],
           "tau": TAU, "epoch_cap": EPOCH_CAP, "seeds": SEEDS, "entries": []}

    def record(label, f, seed=None, selected_epoch=None):
        v = dps(VAL, f, tau=TAU, look_back=LOOK_BACK, train_frac=1e-6, unit=unit)
        t_ = dps(TEST, f, tau=TAU, look_back=LOOK_BACK, train_frac=1e-6, unit=unit)
        key = "persistence" if f is None else label.split()[0]
        e = energy_report(key if key in MACS else "persistence",
                          t_["rho"], t_["n_steps"], t_["n_init_reports"])
        # independent sender/receiver confirmation of the bound, on the test span
        vr = verify(TEST, (lambda ff: (lambda: ff))(f), tau=TAU, look_back=LOOK_BACK,
                    train_frac=1e-6, unit=unit, strict=True)
        res["entries"].append({"predictor": label, "seed": seed,
                               "selected_epoch": selected_epoch,
                               "val_rho": v["rho"], "test_rho": t_["rho"],
                               "test_mean_err": t_["mean_err"], "test_max_err": t_["max_err"],
                               "test_rmse": t_["rmse"], "test_mae": t_["mean_err"],
                               "test_nonfinite": t_["n_nonfinite_forecasts"],
                               "receiver_verified": vr["bound_holds"], "energy": e})
        if out:
            json.dump(res, open(out, "w"), indent=1)        # incremental save
        print(f"  {label:<26} val={v['rho']:.3f} test={t_['rho']:.3f} "
              f"max={t_['max_err']:.4f} E_tot={e['E_total']:.4g} J", flush=True)

    print(f"[{name}] T={T} split tr<{tr} val<{va} test>={va}", flush=True)
    # NOTE: this is ordinary least-squares line-fit extrapolation over the window,
    # NOT a fitted autoregressive model with estimated coefficients. Named accordingly.
    record("persistence", None)
    _g, _k = select_ar(VAL, TAU, LOOK_BACK, unit)
    res["lin_k"] = _k
    MACS["linear_extrap(k=%d)" % _k] = _k
    record("linear_extrap(k=%d)" % _k, _g)
    _f, _a = select_expsm(VAL, TAU, LOOK_BACK, unit)
    res["expsm_alpha"] = _a
    record("expsm(a=%.2f)" % _a, _f)

    if do_learned:
        Xtr, Ytr = windows(A, A, LOOK_BACK, tr, rng)
        mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
        my, sy = float(Ytr.mean()), float(Ytr.std() + 1e-8)
        for kind in ("lightweight", "cnn_lstm"):
            for sd in SEEDS:
                m = build(kind, sd); best = (-1, None)
                for ep in range(CKPT_EVERY, EPOCH_CAP + 1, CKPT_EVERY):
                    m.fit((Xtr - mx) / sx if kind == "lightweight"
                          else ((Xtr - mx) / sx)[..., None],
                          (Ytr - my) / sy, epochs=CKPT_EVERY, batch_size=64,
                          verbose=0, validation_split=0.15)
                    f = wrap(m, kind, mx, sx, my, sy)
                    r = dps(VAL, f, tau=TAU, look_back=LOOK_BACK,
                            train_frac=1e-6, unit=unit)["rho"]
                    if r > best[0]:                       # selection on VALIDATION
                        best = (r, ep, m.get_weights())
                m.set_weights(best[2])
                record(kind, wrap(m, kind, mx, sx, my, sy), seed=sd, selected_epoch=best[1])

    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


# ---- dataset specification (must be declared; see the guard below) ----------
# The Intel matrix and the 600-row subset used in the submitted manuscript are
# NOT interchangeable, and the difference falls almost entirely in the test span.
# Measured on the 34,478 x 33 matrix, split 50/20/30:
#       span    rows    samples >40C   p99 |step|   open-loop rho @ tau=0.10
#       train  17,239              0       3.90 C               0.961
#       val     6,895              0       0.09 C               0.994
#       test   10,344          1,266       8.59 C               0.885
# The >40 C samples are sensor-failure artifacts (battery depletion) affecting 13
# of 33 motes, and they begin at row 26,112 -- inside the test span, which starts
# at 24,134. Scoring on the full matrix therefore scores partly on hardware
# failure rather than on temperature prediction, and the validation span is an
# unusually quiet period. Both facts must be stated wherever the numbers are.
# Checked: the persistence-vs-linear_extrap ranking reversal is NOT an artifact of
# those spikes. Restricted to the 20 motes that never exceed 40 C, persistence
# still leads on test (0.833 vs 0.721) while linear_extrap leads on validation.
INTEL_SPEC = "time"      # PRIMARY dataset, re-declared 26 Sep 2026 after the provenance repair.
#   "time"  intel_time.npy, 25,127 x 30: TIMESTAMP-indexed 31-s bins, built by
#           prepare_intel.py (sha256 8d406c1e...) from the raw archive (sha256
#           21d15688...). This is the only Intel matrix whose rows are calendar
#           time. Rules R1-R6 in that script; mote inclusion measured over each
#           mote's own active span (full-grid measurement admits only 5 motes).
#   "full"  intel_full.npy, 34,478 x 33 -- WITHDRAWN as a time series. It was
#           reconstructed (99.976% of cells) as an EPOCH-indexed pivot with the
#           same rules, and the Intel epoch counter restarts on every mote
#           reboot (40-190 restarts per mote), so 21.7% of its rows combine
#           readings taken more than a day apart and 15,924 cells average
#           readings from different days. Every result computed on it is
#           development evidence only. The 274 cells (0.024%) that the
#           reconstruction does not reproduce remain unexplained.
#   "full"        34,478 x 33, failure period included; disclose it in the paper.
#   "clean_motes" the 20 motes that never exceed 40 C; a stated exclusion rule.
#   "subset600"   the manuscript's original first 600 rows; matches the submitted
#                 description but leaves ~300 training rows, which is thin.

def load_intel(spec):
    if spec == "time":        return np.load("intel_time.npy")
    if spec not in ("full", "clean_motes", "subset600"):
        raise SystemExit(
        "INTEL_SPEC is not set. This is a protocol decision that changes the "
        "headline suppression number, so it must be recorded before the run, not "
        "chosen afterwards. Set it to 'full', 'clean_motes' or 'subset600' and "
        "state the same choice in the manuscript.")
    A = np.load("intel_full.npy")
    if spec == "full":        return A
    if spec == "clean_motes": return A[:, ~(A > 40).any(0)]
    return A[:600]


if __name__ == "__main__":
    A = load_intel(INTEL_SPEC)
    print(f"[data] intel spec={INTEL_SPEC} shape={A.shape}", flush=True)
    run_dataset(A, f"intel_temperature[{INTEL_SPEC}]", unit=1.0,
                do_learned=True, out="bench_intel.json")
