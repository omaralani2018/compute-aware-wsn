"""
Matched comparison with two published dual-prediction schemes, run under the
protocol of this paper (frozen_benchmark.py / dual_prediction_fix.py).

  LMS   Santini & Roemer [13]: LMS adaptive filter of length N, step size mu,
        weights adapted only on transmitted rounds, w[0] = 0.
  AM-DR Fathy et al. [25]: convex mix lambda of a fast moving average over the
        last w_f-1 shared values and a slow mean over the shared values since
        the last transmission; lambda adapted on transmitted rounds with rate
        alpha, lambda[0] = 0.

Both are taken from the authors' released MATLAB code (github.com/YasminFathy/
AMDRIoT, baseline/dual_prediction.m and myapproach_AMDR/AMDR_dual_prediction.m);
ports_faithful.py reproduces those scripts line by line and port_validation.json
checks them against the published numbers. Here the same update equations are
written as stateful predictors so that they run inside this paper's protocol.

Two transmission rules are evaluated for each scheme:
  "authors": suppress only if the last W errors are all strictly below tau
             (W = N for LMS, W = w_f for AM-DR), as in the released code;
  "ours":    suppress if the current error is at most tau (Section 3.3).

Protocol, identical to frozen_benchmark.run_dataset:
  split 50/20/30 by row; validation slice A[tr-12:va], test slice A[va-12:];
  the first LOOK_BACK = 12 rows of a slice are the synchronised warm-up
  (transmitted, charged as n_init_reports, not scored); tau = 0.10 physical on
  Intel and 0.10 x training-span std on NAB; every node is its own stream; the
  scheme's state starts from zero at the start of each slice (w[0] = 0 and
  lambda[0] = 0, as in both papers). Hyperparameters are selected on validation
  suppression, ties going to the first grid entry. Test is scored once.
"""
import numpy as np, json, sys, os

LOOK_BACK = 12
TAU = 0.10
SPLIT = (0.50, 0.70)
E_OP, E_SENSE, E_R = 1e-9, 1e-7, 3.8e-4

LMS_N_GRID = [4, 5, 6, 8, 10]                        # Santini: N = 4..10
LMS_C_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]  # mu = c / Ex_train, Ex = mean(x^2);
# c = 1 is Santini & Roemer's convergence bound (their Eq. 5, 0 <= mu <= 1/Ex).
# The first run stopped at c = 0.1 and validation selected that edge on four streams,
# so the grid was extended to the bound; a second run found the lower edge (1e-4)
# selected on twitter_aapl, so the grid was extended down to 1e-6 (both are
# validation-side decisions, disclosed).
AMDR_WIN_GRID = [(4, 8), (5, 10)]                    # (w_f, w_s) published by Fathy et al.
AMDR_C_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]  # alpha = c / var_train
# The first run stopped at c = 1 and validation selected that edge on three streams;
# extended by two decades (validation-side decision, disclosed). The released code
# leaves lambda unclipped, and so does this port.

# MAC charge per round (multiply-accumulates; additions alone are not charged,
# the same convention as the cheap arms):
#   LMS: N for the prediction and N+1 for the weight update = 2N+1, the count
#        given by Santini & Roemer; charged every round (an upper bound, since
#        the update runs only on transmitted rounds).
#   AM-DR: the fast mean as w_f-1 weighted accumulations, the slow running mean
#        (1), the convex mix (2) and the lambda update (2) = w_f + 4.
def macs_lms(N): return 2 * N + 1
def macs_amdr(w_f): return w_f + 4


class LMS:
    def __init__(self, n_nodes, N, mu, warm):
        # warm-up rows: the first N are the released code's initialisation
        # (x_e = x, no update); rows N..LOOK_BACK-1 are transmitted rounds and go
        # through the transmit branch, exactly as in the released loop.
        self.N, self.mu = N, mu
        W = np.array(warm, float)
        self.w = np.zeros((n_nodes, N))
        self.hist = W[:N].T.copy()                       # (n_nodes, N) oldest..newest
        for k in range(N, W.shape[0]):
            e = W[k] - (self.w * self.hist).sum(1)
            self.w = self.w + mu * self.hist * e[:, None]
            self.hist = np.concatenate([self.hist[:, 1:], W[k][:, None]], 1)
    def predict(self):
        return (self.w * self.hist).sum(1)
    def update(self, sent, value, pred):
        e = value - pred                                 # zero where suppressed
        # w(k+1,i) = w(k,i) + mu * x_e(k-N+i-1) * e(k), only on transmitted rounds
        self.w = np.where(sent[:, None], self.w + self.mu * self.hist * e[:, None], self.w)
        self.hist = np.concatenate([self.hist[:, 1:], value[:, None]], 1)


class AMDR:
    def __init__(self, n_nodes, w_f, w_s, alpha, warm):
        # warm-up rows are the released code's initialisation (x_e = x, start = 1);
        # w_s <= LOOK_BACK, so the warm-up covers it. Rows w_s..LOOK_BACK-1 of the
        # warm-up are transmitted rounds and go through the transmit branch below.
        self.w_f, self.alpha = w_f, alpha
        self.lam = np.zeros(n_nodes)
        W = np.array(warm, float)                        # (LOOK_BACK, n_nodes)
        self.buf = W[-(w_f - 1):].T.copy()               # last w_f-1 shared values
        # replay the transmit branch for warm-up rows k = w_s .. LOOK_BACK-1
        self.s_sum = W[:w_s].sum(0); self.s_cnt = np.full(n_nodes, float(w_s))
        for k in range(w_s, W.shape[0]):
            maf = W[k - w_f + 1:k].mean(0); mas = self.s_sum / self.s_cnt
            y = self.lam * maf + (1 - self.lam) * mas
            self.lam = self.lam + alpha * (W[k] - y) * (maf - mas)
            self.s_sum = W[k].copy(); self.s_cnt = np.ones(n_nodes)   # start = k
    def predict(self):
        self.maf = self.buf.mean(1); self.mas = self.s_sum / self.s_cnt
        return self.lam * self.maf + (1 - self.lam) * self.mas
    def update(self, sent, value, pred):
        e = value - pred
        self.lam = np.where(sent, self.lam + self.alpha * e * (self.maf - self.mas), self.lam)
        self.s_sum = np.where(sent, value, self.s_sum + value)   # start = k on transmit
        self.s_cnt = np.where(sent, 1.0, self.s_cnt + 1.0)
        self.buf = np.concatenate([self.buf[:, 1:], value[:, None]], 1)


class Persistence:                                       # sanity check against dps()
    def __init__(self, n_nodes, warm): self.last = np.array(warm[-1], float)
    def predict(self): return self.last.copy()
    def update(self, sent, value, pred): self.last = value.copy()


def run_scheme(X, make, rule, W, tau=TAU, unit=1.0, return_packets=False):
    """Sender side. make(warm) -> scheme. Returns the dps()-style summary."""
    X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X
    T, n = X.shape; t0 = LOOK_BACK
    s = make(X[:t0])
    ehist = np.full((n, W), np.inf)                      # forces sends until W errors are known
    n_tx = 0; errs = []; recon = X.copy(); packets = np.zeros((T, n), bool)
    for t in range(t0, T):
        pred = s.predict()
        err = np.abs(X[t] - pred) / unit
        if rule == "ours":
            sup = err <= tau
        else:
            ehist = np.concatenate([ehist[:, 1:], err[:, None]], 1)
            sup = (ehist < tau).all(1)
        sent = ~sup
        value = np.where(sent, X[t], pred)
        s.update(sent, value, pred)
        recon[t] = value; packets[t] = sent
        errs.append(np.where(sup, err * unit, 0.0)); n_tx += int(sent.sum())
    errs = np.concatenate(errs); n_steps = (T - t0) * n
    out = {"rho": 1 - n_tx / n_steps, "mean_err": float(errs.mean()),
           "rmse": float(np.sqrt((errs ** 2).mean())), "max_err": float(errs.max()),
           "n_steps": n_steps, "n_tx": n_tx, "n_init_reports": t0 * n}
    if return_packets: out["_packets"], out["_recon"] = packets, recon
    return out


def receiver(X, make, packets, tau, unit):
    """Independent receiver: its own scheme instance, fed only the packets."""
    X = np.asarray(X, float); X = X[:, None] if X.ndim == 1 else X
    T, n = X.shape; t0 = LOOK_BACK
    r = make(X[:t0]); recon = X.copy()
    for t in range(t0, T):
        pred = r.predict(); sent = packets[t]
        value = np.where(sent, X[t], pred)               # X[t] used only where a packet arrived
        r.update(sent, value, pred); recon[t] = value
    return recon


def evaluate(Xslice, make, rule, W, unit, verify=False):
    o = run_scheme(Xslice, make, rule, W, unit=unit, return_packets=verify)
    if verify:
        rec = receiver(Xslice, make, o["_packets"], TAU, unit)
        Xs = np.asarray(Xslice, float); Xs = Xs[:, None] if Xs.ndim == 1 else Xs
        o["receiver_identical"] = bool(np.array_equal(rec, o["_recon"]))
        o["receiver_max_err"] = float(np.abs(rec[LOOK_BACK:] - Xs[LOOK_BACK:]).max())
        o["bound_holds"] = bool(o["receiver_max_err"] <= TAU * unit + 1e-12)
        del o["_packets"], o["_recon"]
    return o


def select_and_test(A, unit, scheme, rule):
    T = A.shape[0]; tr, va = int(SPLIT[0] * T), int(SPLIT[1] * T)
    VAL, TEST = A[tr - LOOK_BACK:va], A[va - LOOK_BACK:]
    TRAIN = A[:tr]
    ex = float((TRAIN ** 2).mean()); var = float(TRAIN.var())
    grid = []
    if scheme == "LMS":
        for N in LMS_N_GRID:
            for c in LMS_C_GRID:
                grid.append(({"N": N, "c": c, "mu": c / ex},
                             (lambda N=N, mu=c / ex: (lambda warm: LMS(A.shape[1], N, mu, warm))), N, macs_lms(N)))
    else:
        for (wf, ws) in AMDR_WIN_GRID:
            for c in AMDR_C_GRID:
                a = c / var
                grid.append(({"w_f": wf, "w_s": ws, "c": c, "alpha": a},
                             (lambda wf=wf, ws=ws, a=a: (lambda warm: AMDR(A.shape[1], wf, ws, a, warm))), wf, macs_amdr(wf)))
    best = None; val_table = []
    for params, mk, W, macs in grid:
        v = evaluate(VAL, mk(), rule, W, unit)["rho"]
        val_table.append([params, v])
        if best is None or v > best[0]:
            best = (v, params, mk, W, macs)
    v, params, mk, W, macs = best
    t = evaluate(TEST, mk(), rule, W, unit, verify=True)
    return {"scheme": scheme, "rule": rule, "selected": params, "macs": macs,
            "val_rho": v, "test": t, "val_grid": val_table}


def published_settings_intel(A):
    """The authors' own settings, physical units, Intel only (tau = 0.10)."""
    T = A.shape[0]; va = int(SPLIT[1] * T); TEST = A[va - LOOK_BACK:]
    out = {}
    for rule in ("authors", "ours"):
        out[f"LMS_N5_mu1e-5_{rule}"] = evaluate(TEST, lambda w: LMS(A.shape[1], 5, 1e-5, w), rule, 5, 1.0, verify=True)
        out[f"AMDR_wf5_ws10_a1e-7_{rule}"] = evaluate(TEST, lambda w: AMDR(A.shape[1], 5, 10, 1e-7, w), rule, 5, 1.0, verify=True)
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    name = sys.argv[1]
    if name == "intel_time":
        A = np.load(sys.argv[2] if len(sys.argv) > 2 else "intel_time.npy"); unit = 1.0
    else:
        import pandas as pd
        from nab_data import NAB
        path = NAB[name]
        v = pd.read_csv(path)["value"].to_numpy(float); A = v[:, None]
        unit = float(A[:int(SPLIT[0] * len(A))].std())
    res = {"dataset": name, "unit": unit, "T": int(A.shape[0]), "N": int(A.shape[1]),
           "tau": TAU, "runs": []}
    for scheme in ("LMS", "AMDR"):
        for rule in ("authors", "ours"):
            r = select_and_test(A, unit, scheme, rule); res["runs"].append(r)
            print(f"[{name}] {scheme:4s} {rule:7s} sel={r['selected']} val={r['val_rho']:.4f} "
                  f"test={r['test']['rho']:.4f} max={r['test']['max_err']:.4g} "
                  f"recv={r['test']['receiver_identical']}/{r['test']['bound_holds']}", flush=True)
    if name == "intel_time":
        res["published_settings"] = published_settings_intel(A)
    json.dump(res, open(f"matched_{name}.json", "w"), indent=1)
