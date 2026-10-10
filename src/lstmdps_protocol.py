"""
LSTM-DPS [18] under this paper's protocol, Intel matrix (intel_time.npy).

Per mote (each column is its own stream, as in the released code, which trains
one model per mote):
  historical data = the last 24 h of the TRAINING span (the released code
      trains on the day before deployment; the training span is the only data
      this protocol allows a predictor to learn from);
  test slice = A[va-12:], the same slice every other arm is scored on; rows
      0..11 are the synchronised warm-up and are not scored;
  the released sequence then runs on the test slice unchanged: 3-reading
      initialisation, 20-step fixed-weight LMS phase, refit on that aggregate
      list, LSTM phase to the end; transmission when |x - x_hat| >= tau
      (the released rule, "< e_max" suppresses), tau = 0.10 C;
  employ_update_policy = False, as in every saved Intel experiment.
An independent receiver reconstructs the series from the packets alone with
the deployed model (in LSTM-DPS the fog node trains and ships the model).
"""
import numpy as np, pandas as pd, json, sys, time
from lstmdps_port import NumpyLSTMDPSModel, LMS_W
import lstmdps_train as lt

TAU = 0.10; LOOK_BACK = 12; SPLIT = (0.50, 0.70)


def sender(x, model_after_refit_fn, tau):
    """Runs the released sequence; model_after_refit_fn(agg_list) -> NumpyLSTMDPSModel."""
    x = np.asarray(x, float)
    lst = [x[0]] + [x[i + 1] - x[i] for i in range(3)]
    temp = float(np.sum(lst)); agg = list(lst[1:])
    tx = np.zeros(len(x), bool); tx[:4] = True; recon = x.copy(); recon[:4] = x[:4]
    def step(i, y_pred):
        nonlocal temp
        if abs(x[i] - (temp + y_pred)) < tau:
            agg.append(y_pred); temp += y_pred
        else:
            agg.append(x[i] - temp); temp = x[i]; tx[i] = True
        recon[i] = temp
    for i in range(4, 24):
        step(i, float(LMS_W @ np.array(agg[-3:])))
    model = model_after_refit_fn(list(agg))
    for i in range(24, len(x)):
        step(i, float(model.predict(np.array(agg[-3:])[None, :])[0]))
    return tx, recon, model


def receiver(x, tx, model):
    """Sees x[i] only where tx[i]; same fixed LMS weights and the shipped model."""
    temp = x[0]; agg = []
    for i in range(1, 4):                       # initialisation readings are transmitted
        agg.append(x[i] - temp); temp = x[i]
    rec = np.empty(len(x)); rec[:4] = x[:4]
    for i in range(4, len(x)):
        y_pred = float(LMS_W @ np.array(agg[-3:])) if i < 24 else float(model.predict(np.array(agg[-3:])[None, :])[0])
        if tx[i]: agg.append(x[i] - temp); temp = x[i]
        else: agg.append(y_pred); temp += y_pred
        rec[i] = temp
    return rec


def run_mote(A, rows_ts, j, seed, tau=TAU):
    T = A.shape[0]; tr, va = int(SPLIT[0] * T), int(SPLIT[1] * T)
    ts = pd.to_datetime(rows_ts)
    t_end = ts[tr]                               # first validation row
    hist_idx = np.where((ts >= t_end - pd.Timedelta(hours=24)) & (ts < t_end))[0]
    historical = A[hist_idx, j]
    t0 = time.time()
    keras_model, p = lt.build(historical, seed)
    def after_refit(agg):
        m = lt.refit(keras_model, p, agg)
        nm = NumpyLSTMDPSModel.from_keras(m)
        # the NumPy forward pass must equal Keras on the deployed model
        Z = np.random.default_rng(seed).normal(0, 0.05, (256, 3))
        kz = m(Z[..., None], training=False).numpy()[:, 0]
        after_refit.max_rel = float(np.max(np.abs(nm.predict(Z) - kz)) / (np.max(np.abs(kz)) + 1e-12))
        assert after_refit.max_rel < 1e-4, after_refit.max_rel
        return nm
    x = A[va - LOOK_BACK:, j]
    tx, recon, model = sender(x, after_refit, tau)
    rec = receiver(x, tx, model)
    sc = slice(LOOK_BACK, None)
    err = np.abs(x[sc] - recon[sc])
    return {"mote_col": j, "seed": seed, "arch": p, "macs": model.macs(), "n_params": model.n_params(),
            "n_hist": int(len(historical)), "rho": float(1 - tx[sc].mean()), "n_tx": int(tx[sc].sum()),
            "n_steps": int(len(x) - LOOK_BACK), "max_err": float(err.max()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "receiver_identical": bool(np.array_equal(rec, recon)),
            "bound_holds": bool(np.abs(x[sc] - rec[sc]).max() < tau),
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    A = np.load("intel_time.npy")
    rows = pd.read_csv("intel_time_rows.csv")["bin_start"].values
    seeds = [int(s) for s in sys.argv[1].split(",")]
    cols = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(A.shape[1]))
    out = sys.argv[3] if len(sys.argv) > 3 else "lstmdps_intel_%s.jsonl" % sys.argv[1].replace(",", "_")
    import os
    done = set()
    if os.path.exists(out):
        for l in open(out):
            d = json.loads(l); done.add((d["seed"], d["mote_col"]))
    for s in seeds:
        for j in cols:
            if (s, j) in done:
                continue                                  # resume after an interruption
            r = run_mote(A, rows, j, s)
            open(out, "a").write(json.dumps(r) + "\n")
            print(json.dumps({k: r[k] for k in ("mote_col", "seed", "rho", "macs", "receiver_identical", "bound_holds", "seconds")}), flush=True)
