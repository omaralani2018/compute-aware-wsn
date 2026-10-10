"""
Checks of the three ported schemes (version 1.2).

  1. LMS [13] and AM-DR [25], line-by-line ports (ports_faithful.py), run on the
     authors' own data selection: the readings of each mote dated 2004-03-06 to
     2004-03-09 in raw/data.txt, in file order (as dataset/load_data.m of the
     AMDRIoT repository), e_max = 0.5 C. Gives the transmitted shares compared
     with the published ones (Table S1 of MATCHED_BASELINES.md), and checks that
     AM-DR with the published alpha = 1e-7 transmits on exactly the same rounds
     as with alpha = 0.
  2. Harness equivalence: the stateful predictors of matched_baselines.py,
     under the released rule, against the line-by-line ports given the same
     12-row synchronised warm-up, on all 30 motes of the Intel test slice at
     tau = 0.10 C, for the published settings and for the settings selected on
     validation (read from matched_intel_time.json). Transmission decisions
     must agree on every round.
  3. LSTM-DPS [18]: the 60 released Intel experiments (5 motes x 12 tolerances)
     replayed with the released _refit.h5 models and the released data file,
     through lstmdps_port.run_published; the transmission counts are compared
     with those saved in the released experiment files. For comparison, the
     same series is also run with persistence (the first reading transmitted,
     then a reading is sent when it differs from the last one sent by more
     than the tolerance).

Usage (inside run/):
  python3 validate_ports.py --lstmdps PATH_TO_LSTM-DPS-for-IoT
      needs raw/data.txt, intel_time.npy, matched_intel_time.json, and a clone of
      github.com/Ahmed-Fathalla/LSTM-DPS-for-IoT with data/Intel-DS.rar
      extracted to data/Intel-DS.csv
  -> port_validation.json, lstmdps_port_validation.csv
"""
import numpy as np, pandas as pd, json, sys, os, re, pickle, argparse, warnings
from ports_faithful import lms_dual_prediction_matlab, amdr_dual_prediction_matlab
import matched_baselines as mb

AUTHORS_DATES = ("2004-03-06", "2004-03-09")
AUTHORS_MOTES = [1, 11, 13, 30, 49]
PUBLISHED_TX_PCT = {11: {"LMS": 8, "AMDR": 5}, 30: {"LMS": 18, "AMDR": 13}}   # [25], Figs. 3-4 and 9


def authors_selection(path="raw/data.txt"):
    rows = {}
    for line in open(path):
        p = line.split()
        if len(p) < 5 or not (AUTHORS_DATES[0] <= p[0] <= AUTHORS_DATES[1]):
            continue
        try:
            m = int(p[3]); t = float(p[4])
        except ValueError:
            continue
        rows.setdefault(m, []).append(t)
    return rows


def part1(raw):
    rows = authors_selection(raw); out = {}
    for m in AUTHORS_MOTES:
        x = np.array(rows[m], float)
        lms = lms_dual_prediction_matlab(x, N=5, mu=1e-5, emax=0.5)
        am = amdr_dual_prediction_matlab(x, w_s=10, w_f=5, alpha=1e-7, emax=0.5)
        am0 = amdr_dual_prediction_matlab(x, w_s=10, w_f=5, alpha=0.0, emax=0.5)
        am8 = amdr_dual_prediction_matlab(x, w_s=8, w_f=4, alpha=1e-7, emax=0.5)
        am8_12 = amdr_dual_prediction_matlab(x, w_s=8, w_f=4, alpha=1.2e-7, emax=0.5)
        am8_0 = amdr_dual_prediction_matlab(x, w_s=8, w_f=4, alpha=0.0, emax=0.5)
        lms4 = lms_dual_prediction_matlab(x, N=4, mu=1e-5, emax=0.5)
        def sink_err(r):            # reconstruction error on suppressed rounds
            return float(np.abs(x - r["x_e"]).max())
        out[str(m)] = {
            "n": int(len(x)),
            "LMS_N5_mu1e-5_tx_pct": round(100 * lms["tx"].mean(), 2),
            "LMS_N4_mu1e-5_tx_pct": round(100 * lms4["tx"].mean(), 2),
            "AMDR_ws10_wf5_a1e-7_tx_pct": round(100 * am["tx"].mean(), 2),
            "AMDR_ws8_wf4_a1e-7_tx_pct": round(100 * am8["tx"].mean(), 2),
            "AMDR_ws8_wf4_a1.2e-7_tx_pct": round(100 * am8_12["tx"].mean(), 2),
            "AMDR_tx_identical_with_alpha0": bool(np.array_equal(am["tx"], am0["tx"])
                                                  and np.array_equal(am8["tx"], am8_0["tx"])
                                                  and np.array_equal(am8_12["tx"], am8_0["tx"])),
            "max_sink_err_LMS": sink_err(lms),
            "max_sink_err_AMDR": sink_err(am),
            "published_tx_pct": PUBLISHED_TX_PCT.get(m),
        }
    return out


# ---- 2. harness equivalence -------------------------------------------------
# The line-by-line ports with their initialisation extended to the protocol's
# synchronised warm-up of LOOK_BACK rows. Two changes, both marked "warm-up":
# rounds before LOOK_BACK always take the transmit branch, and their errors are
# recorded as unknown (inf), as the harness does, so that the released rule
# cannot suppress until W errors of the scored span are known.
def lms_warm(x, N, mu, emax, warm=mb.LOOK_BACK):
    L = len(x); w = np.zeros((L + 1, N)); x_e = np.zeros(L); e = np.zeros(L); tx = np.zeros(L, bool)
    tx[:N] = True; x_e[:N] = x[:N]; e[:N] = np.inf
    for k in range(N, L):
        y = np.dot(w[k], x_e[k - N:k]); e_k = x[k] - y
        e[k] = np.inf if k < warm else abs(e_k)                       # warm-up
        if k >= warm and np.all(e[k - N + 1:k + 1] < emax):          # warm-up
            x_e[k] = y; w[k + 1] = w[k]
        else:
            tx[k] = True; x_e[k] = x[k]; w[k + 1] = w[k] + mu * x_e[k - N:k] * e_k
    return tx


def amdr_warm(x, w_s, w_f, alpha, emax, warm=mb.LOOK_BACK):
    L = len(x); lam = 0.0; x_e = np.zeros(L); e = np.zeros(L); tx = np.zeros(L, bool)
    x_e[:w_s] = x[:w_s]; tx[:w_s] = True; e[:w_s] = np.inf; start = 0
    for k in range(w_s, L):
        maf = x_e[k - w_f + 1:k].mean(); mas = x_e[start:k].mean()
        y = lam * maf + (1 - lam) * mas; e_k = x[k] - y
        e[k] = np.inf if k < warm else abs(e_k)                       # warm-up
        if k >= warm and np.all(e[k - w_f + 1:k + 1] < emax):        # warm-up
            x_e[k] = y
        else:
            tx[k] = True; x_e[k] = x[k]; lam = lam + alpha * e_k * (maf - mas); start = k
    return tx


def part2():
    A = np.load("intel_time.npy"); T = A.shape[0]
    tr, va = int(mb.SPLIT[0] * T), int(mb.SPLIT[1] * T)
    X = A[va - mb.LOOK_BACK:]; ex = float((A[:tr] ** 2).mean()); var = float(A[:tr].var())
    sel = {(r["scheme"], r["rule"]): r["selected"] for r in json.load(open("matched_intel_time.json"))["runs"]}
    s_l, s_a = sel[("LMS", "authors")], sel[("AMDR", "authors")]
    configs = [("LMS", "published", 5, 1e-5), ("LMS", "selected", s_l["N"], s_l["c"] / ex),
               ("AMDR", "published", (5, 10), 1e-7), ("AMDR", "selected", (s_a["w_f"], s_a["w_s"]), s_a["c"] / var)]
    out = {"slice": "Intel test slice A[va-12:], 30 motes, tau = 0.10 C, released rule", "configs": []}
    for scheme, kind, p, rate in configs:
        if scheme == "LMS":
            h = mb.run_scheme(X, lambda w: mb.LMS(A.shape[1], p, rate, w), "authors", p, return_packets=True)["_packets"]
            f = np.stack([lms_warm(X[:, j], p, rate, mb.TAU) for j in range(A.shape[1])], 1)
            params = {"N": p, "mu": rate}
        else:
            wf, ws = p
            h = mb.run_scheme(X, lambda w: mb.AMDR(A.shape[1], wf, ws, rate, w), "authors", wf, return_packets=True)["_packets"]
            f = np.stack([amdr_warm(X[:, j], ws, wf, rate, mb.TAU) for j in range(A.shape[1])], 1)
            params = {"w_f": wf, "w_s": ws, "alpha": rate}
        d = int((h[mb.LOOK_BACK:] != f[mb.LOOK_BACK:]).sum())
        out["configs"].append({"scheme": scheme, "settings": kind, "params": params,
                               "rounds_compared": int(h[mb.LOOK_BACK:].size), "disagreements": d})
    return out


# ---- 3. LSTM-DPS replay -----------------------------------------------------
def persistence_pct(x, tau):
    last, tx = x[0], 1
    for v in x[1:]:
        if abs(v - last) > tau:
            last, tx = v, tx + 1
    return round(100 * (1 - tx / len(x)), 2)


def part3(repo, out_csv):
    from lstmdps_port import NumpyLSTMDPSModel, run_published
    sys.path.insert(0, repo)                       # the saved experiments pickle the authors' classes
    warnings.filterwarnings("ignore")              # NumPy deprecation notices raised while unpickling
    df = pd.read_csv(os.path.join(repo, "data", "Intel-DS.csv"))       # utils/_utils.read_Intel_data
    df["date_modified"] = df["date"].apply(lambda x: str(x).replace("-", ""))
    df["time"] = df["time"].apply(lambda x: x[:8])
    df["time_stamp"] = pd.to_datetime(df["date_modified"] + ":" + df["time"], format="%Y%m%d:%H:%M:%S")
    df.sort_values(by="time_stamp", inplace=True)
    rows = []
    for m in AUTHORS_MOTES:
        d = df[df.moteid == m][["temperature", "time_stamp", "date"]].copy()   # load_mote_data
        d.sort_values(by="time_stamp", inplace=True)
        x = d[(d.date >= AUTHORS_DATES[0]) & (d.date <= AUTHORS_DATES[1])]["temperature"].values
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            ex = pickle.load(open(os.path.join(repo, "Experiments", "Exp Intel-DS Mote_%d Epsilon_%.1f.pkl" % (m, eps)), "rb"))
            pub = float(re.search(r"Reduc:\s*([0-9.]+)%", ex.experimental_results).group(1))
            miss_pub = int(ex.total_miss_pred_count)
            n_pub = int(sum(v["No_of_observations"] for v in ex.phase.values()))
            h5 = os.path.join(repo, "models", "LSTM_DPS_model Intel-DS Mote_%d Epsilon_%.1f_refit.h5" % (m, eps))
            if len(x) == 0:
                rows.append([m, eps, pub, "", "", miss_pub, n_pub, 0, "", "no readings of this mote in the released data file"])
                continue
            r = run_published(x, NumpyLSTMDPSModel.from_h5(h5), eps)
            note = "" if len(x) == n_pub else "released data file has %d readings, saved run used %d" % (len(x), n_pub)
            rows.append([m, eps, pub, round(r["reduction_pct"], 2), r["miss"], miss_pub, n_pub, len(x),
                         persistence_pct(x, eps), note])
    cols = ["mote", "eps", "published_pct", "port_pct", "miss_port", "miss_pub", "n_pub", "n_port",
            "persistence_pct", "note"]
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    exact = [r for r in rows if r[4] != "" and r[4] == r[5] and r[6] == r[7]]
    return {"experiments": len(rows), "replayed": sum(r[4] != "" for r in rows),
            "same_data_length": sum(r[4] != "" and r[6] == r[7] for r in rows),
            "transmission_count_exact": len(exact),
            "exact_motes": sorted({r[0] for r in exact})}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/data.txt")
    ap.add_argument("--lstmdps", default=None, help="path to a clone of LSTM-DPS-for-IoT")
    a = ap.parse_args()
    res = {"source_lms_amdr": "github.com/YasminFathy/AMDRIoT (baseline/dual_prediction.m, myapproach_AMDR/AMDR_dual_prediction.m)",
           "selection": "raw/data.txt rows with date in [2004-03-06, 2004-03-09], file order, as dataset/load_data.m",
           "emax": 0.5}
    res["motes"] = part1(a.raw)
    for m, v in res["motes"].items():
        print("mote", m, {k: v[k] for k in v if "pct" in k or "alpha0" in k})
    res["harness_equivalence"] = part2()
    for c in res["harness_equivalence"]["configs"]:
        print("equivalence", c["scheme"], c["settings"], c["params"], "disagreements:", c["disagreements"], "of", c["rounds_compared"])
    if a.lstmdps:
        res["lstmdps_replay"] = part3(a.lstmdps, "lstmdps_port_validation.csv")
        print("LSTM-DPS replay", res["lstmdps_replay"])
    json.dump(res, open("port_validation.json", "w"), indent=1)
