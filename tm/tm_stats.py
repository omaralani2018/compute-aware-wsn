"""Analysis of TM_PROTOCOL.md section 4. Reads tm_runs/tm_<stream>_s<seed>.json and ms_numbers.json
(validation-selected low-cost reference at tau = 0.10); writes tm_summary.json.

Primary, per stream, paired over seeds 0-9: d = test suppression(C) - test suppression(B+). Estimand: the
mean d. Test: two-sided sign-flip permutation test of the mean (all 2^10 assignments), computed on the
integer differences in transmitted readings (both arms score the same test steps), so that ties are exact.
Bonferroni 0.05/11. Outcome: "C higher", "B+ higher" or "inconclusive". The test assumes that, under the
null, the paired seed differences are independent and symmetric about zero (sign-exchangeable); the
enumeration is exact under that assumption. A zero difference contributes nothing to any sign assignment;
ties between assignments are counted exactly (integers). Interval: 95% percentile bootstrap of the mean d
over seeds (10,000 resamples, numpy default_rng(0) started afresh for each stream, numpy's default linear
percentile), descriptive. An incomplete stream gets mean d and interval only (no p, outcome inconclusive,
energy labels "incomplete"). A seed that is missing or fails a
receiver check in either arm is excluded from both arms (paired). A stream with fewer than 10 valid
seeds (missing, or a failed receiver check in either arm) is "incomplete": descriptive only, outcome
inconclusive.

Descriptive (section 4): each arm's net saving per node-round against the reference,
g = e_r*(rho - rho_ref) - (208 - M_ref)*E_op at e_r = 3.8e-4 J and E_op = 1 nJ, its mean, and a label from the
same exact test at the same threshold ("saves", "costs", "ns"), computed in exact integer units; for
complete streams only. This label is not the main-study verdict (Wilcoxon over 20 seeds at 0.05/22). The
same label for the phase-1 checkpoints of the same 10 seeds (the main-study models) is reported as context,
provided their receiver checks passed. Every run must carry the same code hashes, versions and settings, the
same input-data hash within a stream, and code hashes listed in TM_FROZEN.sha256 (in the working directory)
and equal to it; only the synthetic check may run without that file, by setting TM_ALLOW_UNFROZEN=1.
Usage: python3 tm_stats.py [runs_dir] [ms_numbers.json]"""
import json, os, sys, numpy as np
STREAMS = ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly", "rds_cpu_util",
           "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]
SEEDS = list(range(10)); ALPHA = 0.05 / len(STREAMS)
E_R_1E13, E_OP_1E13, MAC = 3_800_000_000, 10_000, 208          # e_r and E_op times 1e13, exact integers
RUNS = sys.argv[1] if len(sys.argv) > 1 else "tm_runs"
MS = sys.argv[2] if len(sys.argv) > 2 else "ms_numbers.json"
SETTINGS = dict(LOOK_BACK=12, TAU=0.1, SPLIT=[0.5, 0.7], EPOCH_CAP=200, CKPT_EVERY=25, MAX_WINDOWS=4000,
                threads={"OMP_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"})
FIRST = {}                                                     # code hashes and versions of the first run read
FROZEN = {}                                                    # file -> SHA-256 from TM_FROZEN.sha256, if present
assert os.path.exists("TM_FROZEN.sha256") or os.environ.get("TM_ALLOW_UNFROZEN") == "1", "TM_FROZEN.sha256 missing"
if os.path.exists("TM_FROZEN.sha256"):
    for line in open("TM_FROZEN.sha256"):
        if line.strip(): h, f = line.split(None, 1); FROZEN[os.path.basename(os.path.normpath(f.strip().lstrip("*")))] = h
assert FROZEN or os.environ.get("TM_ALLOW_UNFROZEN") == "1", "TM_FROZEN.sha256 missing or empty"


def p_perm(x):
    """Exact two-sided sign-flip p of the mean for an integer vector."""
    x = np.asarray(x, dtype=np.int64); n = len(x)
    if n == 0 or not np.any(x != 0): return 1.0
    s = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int64) * 2 - 1
    return float(np.mean(np.abs(s @ x) >= abs(int(x.sum()))))


def boot_ci(x, n_steps, B=10_000):
    """95% percentile bootstrap interval of the mean of x / n_steps over seeds (resampled with replacement)."""
    x = np.asarray(x, dtype=np.int64); idx = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
    m = x[idx].sum(1) / (len(x) * n_steps)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def label(x, p, pos, neg):
    if p >= ALPHA or int(np.sum(x)) == 0: return "inconclusive" if pos != "saves" else "ns"
    return pos if np.sum(x) > 0 else neg


ms = json.load(open(MS))["datasets"]
out = {"protocol": "TM_PROTOCOL.md", "alpha": ALPHA, "frozen_hashes_checked": bool(FROZEN), "streams": {}}
for st in STREAMS:
    R, missing, invalid = [], [], []
    for sd in SEEDS:
        f = os.path.join(RUNS, f"tm_{st}_s{sd}.json")
        if not os.path.exists(f): missing.append(sd); continue
        r = json.load(open(f))
        assert r["stream"] == st and r["seed"] == sd and r["kind"] == "lightweight", f
        assert all(r["settings"][k] == v for k, v in SETTINGS.items()), (f, r["settings"])
        assert r["settings"]["phase2_seed"] == sd + 1000, f
        sig = (json.dumps(r["settings"]["code_sha256"], sort_keys=True), json.dumps(r["versions"], sort_keys=True))
        FIRST.setdefault("sig", sig); assert sig == FIRST["sig"], ("code or versions differ between runs", f)
        FIRST.setdefault(("data", st), r["settings"]["data_sha256"])
        assert r["settings"]["data_sha256"] == FIRST[("data", st)], ("input data differ between seeds", f)
        for cf, h in r["settings"]["code_sha256"].items():
            cf = os.path.basename(os.path.normpath(cf))
            assert not FROZEN or FROZEN.get(cf) == h, ("code missing from or different to TM_FROZEN.sha256", cf, f)
        for arm in ("B+", "C"):
            assert 200 < r["arms"][arm]["selected_epoch"] <= 400, f
            assert r["arms"][arm]["test_n_steps"] == r["phase1"]["test_n_steps"], f
        assert r["arms"]["B+"]["start_state_sha256"] == r["arms"]["C"]["start_state_sha256"] == r["start_state"]["sha256"], ("arm start states differ", f)
        assert r["arms"]["B+"]["updates"] == r["arms"]["C"]["updates"] > 0, ("arm update counts differ", f)
        if not (r["arms"]["B+"]["receiver_verified"] and r["arms"]["C"]["receiver_verified"]): invalid.append(sd); continue
        R.append(r)
    ref = ms[st]["by_tau"]["0.1"]; sel = ref["selected"]
    rho_ref, mac_ref = ref["test"][sel]["rho"], ref["selected_macs"]
    S = dict(n_valid=len(R), seeds_missing=missing, seeds_invalid_receiver=invalid,
             status="complete" if len(R) == len(SEEDS) else "incomplete",
             reference=dict(arm=sel, rho=rho_ref, macs=mac_ref),
             phase1_reproduced=[r["main_study_entry"]["reproduced"] if r.get("main_study_entry") else None for r in R])
    if R:
        n = R[0]["phase1"]["test_n_steps"]
        assert all(r["phase1"]["test_n_steps"] == n for r in R)
        ntx_ref = round((1.0 - rho_ref) * n); assert abs((1.0 - ntx_ref / n) - rho_ref) < 1e-12
        k = np.array([r["arms"]["B+"]["test_n_tx"] - r["arms"]["C"]["test_n_tx"] for r in R], dtype=np.int64)  # C suppresses k more
        p = p_perm(k)
        S.update(test_steps=n, d_mean=float(k.mean() / n), d_ci95_bootstrap=boot_ci(k, n), d_per_seed=(k / n).tolist(),
                 n_zero_differences=int(np.sum(k == 0)), p_d=p if S["status"] == "complete" else None,
                 outcome=label(k, p, "C higher", "B+ higher") if S["status"] == "complete" else "inconclusive")
        S["arms"] = {}
        p1_ok = all(r["phase1"]["receiver_verified"] for r in R)
        for arm in ("phase1", "B+", "C"):
            ntx = np.array([(r["phase1"] if arm == "phase1" else r["arms"][arm])["test_n_tx"] for r in R], dtype=np.int64)
            gx = (ntx_ref - ntx) * E_R_1E13 - (MAC - mac_ref) * E_OP_1E13 * n          # g * n * 1e13, exact
            pg = p_perm(gx)
            S["arms"][arm] = dict(rho_mean=float(1.0 - ntx.mean() / n), dd_mean=float((ntx_ref - ntx).mean() / n),
                                  net_saving_uJ_per_node_round=float(gx.mean() / n / 1e13 * 1e6),
                                  p_net=pg if S["status"] == "complete" else None,
                                  label=("incomplete" if S["status"] != "complete" else
                                         "receiver check failed" if arm == "phase1" and not p1_ok else label(gx, pg, "saves", "costs")),
                                  selected_epochs=[(r["phase1"] if arm == "phase1" else r["arms"][arm])["selected_epoch"] for r in R])
        S["energy_label_differs_B+_C"] = S["arms"]["B+"]["label"] != S["arms"]["C"]["label"]
        S["reconstruction"] = dict(train_closed_loop_rho=float(np.mean([r["reconstruction"]["train_closed_loop_rho"] for r in R])),
                                   windows_changed=float(np.mean([r["reconstruction"]["windows_changed"] for r in R])))
    else:
        S.update(outcome="inconclusive")
    out["streams"][st] = S
json.dump(out, open("tm_summary.json", "w"), indent=1)
for st, S in out["streams"].items():
    print(f"{st:17s} n={S['n_valid']:2d} {S['status']:10s} " +
          (f"d={S['d_mean']:+.4f} [{S['d_ci95_bootstrap'][0]:+.4f}, {S['d_ci95_bootstrap'][1]:+.4f}] p={S['p_d'] if S['p_d'] is None else round(S['p_d'], 4)} {S['outcome']:12s} | phase1 {S['arms']['phase1']['label']:10s} "
           f"B+ {S['arms']['B+']['label']:10s} C {S['arms']['C']['label']}" if S["n_valid"] else "no valid seeds"))
