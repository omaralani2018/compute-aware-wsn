"""Checks of the TM_PROTOCOL.md code on generated data only (no recording is read). Run in a directory
holding the package src/*.py, ms_numbers.json, learned_<stream>.json and the tm_*.py files; it works in a
subfolder (tm_synth_check, removed at the end) and writes nothing else. For speed the epoch budget is cut to
50 (two checkpoints per phase) by setting frozen_benchmark.EPOCH_CAP; the logic is the same.
 1. reconstruct() gives exactly the suppression of the package evaluator (persistence and a trained model).
 2. Phase 1 reproduces run_learned.run for the same seed (selected epoch, validation and test suppression).
 3. With tau_recon = 0 the reconstructed history equals the actual one, and the two arms are identical
    (same checkpoints, validation history and test suppression): the arms differ only through their windows.
 4. With tau = 0.10 the arm-C windows differ, keep positions and targets, and every receiver check passes;
    also for a three-node stream.
 4c. The arms start from the epoch-cap state, not from the restored validation-selected checkpoint: a fresh
    phase-1 training without validation scoring reaches the same state hash; equal start states and update
    counts are recorded for both arms.
 5. tm_stats.py: outcomes, the exact test (with ties and zeros), the bootstrap interval (against an
    explicit-loop computation with a hand-written percentile), incomplete and invalid seeds, unequal start
    states or update counts, the frozen-hash requirement, on constructed result files.
 6. tm_driver.sh: a setup error starts no attempt; completed attempts counted, interrupted attempts not
    counted, a seed whose worker still holds its lock skipped, finished seeds skipped.
Usage: python3 tm_synthetic_check.py"""
import os, sys, json, shutil, subprocess, numpy as np
os.environ["TM_ALLOW_UNFROZEN"] = "1"                 # constructed result files below carry no frozen hashes
import frozen_benchmark as fb
fb.EPOCH_CAP = 50
import run_learned, tm_run
HERE = os.getcwd()
os.makedirs("tm_synth_check", exist_ok=True); os.chdir("tm_synth_check")      # writes nothing in the run directory
P = lambda f: os.path.join(HERE, f)

def stream(rng, T, N):
    t = np.arange(T)[:, None]
    return 20 + 2 * np.sin(2 * np.pi * t / 288 + rng.uniform(0, 6, N)) + np.cumsum(rng.normal(0, 0.03, (T, N)), 0)

rng = np.random.default_rng(42)
A1 = stream(rng, 3000, 1); A3 = stream(rng, 2000, 3); unit = 1.0

# 1. evaluator identity
for X in (A1, A3):
    r_pkg = fb.dps(X, None, tau=fb.TAU, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)["rho"]
    _, r_tm = tm_run.reconstruct(X, lambda W: W[:, -1], fb.TAU, fb.LOOK_BACK, unit)
    assert r_pkg == r_tm, (r_pkg, r_tm)
print("1a reconstruct() == package evaluator (persistence): OK")

# 2. phase 1 reproduces run_learned
for f in ("learned_synthA.json",):
    if os.path.exists(f): os.remove(f)
run_learned.run(A1, "synthA", unit, ["lightweight"], [3], "learned_synthA.json")
res = tm_run.run(A1, "synthA", unit, 3, workdir=".")
assert res["main_study_entry"]["reproduced"], res["main_study_entry"]
print("2  phase 1 reproduces run_learned (seed 3):", res["phase1"]["selected_epoch"], round(res["phase1"]["test_rho"], 4), "OK")
# 1b. reconstruct() == evaluator with the trained model (epoch-200 analogue is checked inside run; here the selected one)
m = fb.build("lightweight", 3)
w = np.load("weights/synthA_lightweight_s3.npz"); m.set_weights([w[k] for k in w.files])
Xtr, Ytr = fb.windows(A1, A1, fb.LOOK_BACK, int(0.5 * len(A1)), np.random.default_rng(0))
stds = (Xtr.mean(0), Xtr.std(0) + 1e-8, float(Ytr.mean()), float(Ytr.std() + 1e-8))
f = run_learned.wrap_fast(m, "lightweight", *stds)
assert fb.dps(A1, f, tau=fb.TAU, look_back=fb.LOOK_BACK, train_frac=1e-6, unit=unit)["rho"] == tm_run.reconstruct(A1, f, fb.TAU, fb.LOOK_BACK, unit)[1]
print("1b reconstruct() == package evaluator (trained model): OK")

# 4. default tau: arm C windows differ, positions and targets kept (asserted inside run), receivers pass
assert res["reconstruction"]["windows_changed"] > 0
for arm in ("B+", "C"): assert res["arms"][arm]["receiver_verified"]
assert res["arms"]["B+"]["selected_epoch"] > fb.EPOCH_CAP and res["arms"]["C"]["selected_epoch"] > fb.EPOCH_CAP
print("4  tau=0.10: windows changed %.3f, train closed-loop rho %.3f, d = %+.4f, receivers OK"
      % (res["reconstruction"]["windows_changed"], res["reconstruction"]["train_closed_loop_rho"], res["d_test_rho_C_minus_Bplus"]))
res3 = tm_run.run(A3, "synthB", unit, 1, workdir=".")
assert res3["reconstruction"]["windows_changed"] > 0 and all(res3["arms"][a]["receiver_verified"] for a in ("B+", "C"))
print("4b three-node stream: OK")

# 4c. the start state is the state after the last phase-1 fit, before any restoring of selected weights
An = 20 + np.random.default_rng(7).normal(0, 0.08, (1500, 1))  # noise stream: phase 1 selects epoch 25 < cap 50
resN = tm_run.run(An, "synthN", unit, 0, workdir=".")              # both arms run (they reload the saved file)
sel = []
for X, sd, r in ((A1, 3, res), (A3, 1, res3), (An, 0, resN)):     # fresh phase-1 training, no validation scoring
    Xt, Yt = fb.windows(X, X, fb.LOOK_BACK, int(fb.SPLIT[0] * len(X)), np.random.default_rng(0))
    mx_, sx_, my_, sy_ = Xt.mean(0), Xt.std(0) + 1e-8, float(Yt.mean()), float(Yt.std() + 1e-8)
    m = fb.build("lightweight", sd)
    for _ in range(fb.CKPT_EVERY, fb.EPOCH_CAP + 1, fb.CKPT_EVERY):
        m.fit((Xt - mx_) / sx_, (Yt - my_) / sy_, epochs=fb.CKPT_EVERY, batch_size=64, verbose=0, validation_split=0.15)
    assert tm_run.state_sha256(m) == r["start_state"]["sha256"] == r["arms"]["B+"]["start_state_sha256"] == r["arms"]["C"]["start_state_sha256"]
    assert r["arms"]["B+"]["updates"] == r["arms"]["C"]["updates"] == r["start_state"]["iterations"]   # same budget as phase 1
    sel.append(r["phase1"]["selected_epoch"])
assert min(sel) < fb.EPOCH_CAP, ("no case where the selected checkpoint precedes the cap", sel)   # makes the check discriminating
print("4c start state = epoch-cap state, not the restored selected checkpoint (phase-1 selected epochs %s), equal in both arms, equal update counts: OK" % sel)

# 3. tau_recon = 0: identical arms
res0 = tm_run.run(A1, "synthA", unit, 3, tau_recon=0.0, workdir=".")
assert res0["reconstruction"]["windows_changed"] == 0.0
a, b = res0["arms"]["B+"], res0["arms"]["C"]
assert a["val_hist"] == b["val_hist"] and a["selected_epoch"] == b["selected_epoch"] and a["test_rho"] == b["test_rho"], (a, b)
assert res0["phase1"]["test_rho"] == res["phase1"]["test_rho"]
print("3  tau_recon=0: arms identical, d = 0; phase 1 repeatable: OK")

# 5. statistics on constructed files (integer transmission counts; n = test steps of the stream)
D = "synth_stats"; shutil.rmtree(D, ignore_errors=True); os.makedirs(D)
ms = json.load(open(P("ms_numbers.json")))["datasets"]
SET = dict(LOOK_BACK=12, TAU=0.1, SPLIT=[0.5, 0.7], EPOCH_CAP=200, CKPT_EVERY=25, MAX_WINDOWS=4000,
           threads={"OMP_NUM_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"},
           code_sha256={"tm_run.py": "x"}, data_sha256="d")
def write(st, sd, kb, kc, ok=True):
    """kb, kc: readings transmitted in arms B+ and C on n test steps."""
    arm = lambda k: dict(test_rho=1 - k / n, test_n_tx=int(k), test_n_steps=n, receiver_verified=ok, selected_epoch=225,
                         start_state_sha256="s", updates=7)
    json.dump(dict(stream=st, seed=sd, kind="lightweight", settings=dict(SET, phase2_seed=sd + 1000),
                   versions={"tensorflow": "2.21.0"}, main_study_entry=None,
                   phase1=dict(test_n_tx=int(kb), test_n_steps=n, selected_epoch=200, receiver_verified=True),
                   start_state=dict(sha256="s"), arms={"B+": arm(kb), "C": arm(kc)},
                   reconstruction=dict(train_closed_loop_rho=0.5, windows_changed=0.5)), open(f"{D}/tm_{st}_s{sd}.json", "w"))
ties = [3, 2, 2, 1, 1, 1, 1, 1, 1, -1]                                   # exact p = 16/1024: inconclusive
for st in ["intel_time", "taxi", "traffic", "cpu", "machine_temp", "grok_asg_anomaly",
           "rds_cpu_util", "iio_networkin", "twitter_aapl", "exchange_cpc", "occupancy"]:
    L = json.load(open(P(f"learned_{st}.json"))); va = L["split"][1]
    n = (L["T"] - va) * (30 if st == "intel_time" else 1)                       # test steps of the stream
    k0 = n // 2
    for sd in range(10):
        if st == "intel_time": write(st, sd, k0, k0 - 100 - sd)                 # C higher on every seed: p = 2/1024
        elif st == "taxi": write(st, sd, k0, k0 + 200)                         # B+ higher
        elif st == "traffic": write(st, sd, k0, k0 - (100 if sd % 2 else -100))  # inconclusive, p = 1
        elif st == "cpu" and sd == 9: continue                                  # missing -> incomplete
        elif st == "machine_temp": write(st, sd, k0, k0 - 100, ok=(sd != 4))   # failed receiver -> incomplete
        elif st == "grok_asg_anomaly": write(st, sd, k0, k0 - ties[sd])         # ties: p = 16/1024
        elif st == "rds_cpu_util": write(st, sd, k0, k0 - (0 if sd == 5 else 1))  # nine +1 and one zero: p = 4/1024
        else: write(st, sd, k0, k0)                                             # no difference
out = subprocess.run([sys.executable, P("tm_stats.py"), D, P("ms_numbers.json")], capture_output=True, text=True)
assert out.returncode == 0, out.stderr
S = json.load(open("tm_summary.json"))["streams"]
assert S["intel_time"]["outcome"] == "C higher" and S["intel_time"]["p_d"] == 2 / 1024
assert S["taxi"]["outcome"] == "B+ higher" and S["traffic"]["outcome"] == "inconclusive" and S["traffic"]["p_d"] == 1.0
assert S["grok_asg_anomaly"]["p_d"] == 16 / 1024 and S["grok_asg_anomaly"]["outcome"] == "inconclusive"
def boot_ref(k, n):          # explicit loop over the specified resamples, hand-written linear percentile
    idx = np.random.default_rng(0).integers(0, len(k), (10_000, len(k)))
    means = sorted(sum(int(k[j]) for j in row) / (len(k) * n) for row in idx)
    def pct(q):
        h = q / 100 * (len(means) - 1); i = int(h)
        return means[i] + (h - i) * (means[min(i + 1, len(means) - 1)] - means[i])
    return [pct(2.5), pct(97.5)]
for st, k in (("intel_time", [100 + sd for sd in range(10)]), ("grok_asg_anomaly", ties), ("traffic", [100, -100] * 5)):
    n_st = S[st]["test_steps"]
    assert np.allclose(S[st]["d_ci95_bootstrap"], boot_ref(k, n_st), rtol=0, atol=1e-15), (st, S[st]["d_ci95_bootstrap"])
lo, hi = S["intel_time"]["d_ci95_bootstrap"]; assert 100 / S["intel_time"]["test_steps"] <= lo <= hi <= 109 / S["intel_time"]["test_steps"]
assert S["occupancy"]["d_ci95_bootstrap"] == [0.0, 0.0] and S["occupancy"]["n_zero_differences"] == 10
z = json.load(open(f"{D}/tm_twitter_aapl_s0.json")); z["arms"]["C"]["start_state_sha256"] = "other"     # unequal start states
json.dump(z, open(f"{D}/tm_twitter_aapl_s0.json", "w"))
assert subprocess.run([sys.executable, P("tm_stats.py"), D, P("ms_numbers.json")], capture_output=True, text=True).returncode != 0
z["arms"]["C"]["start_state_sha256"] = "s"; z["arms"]["C"]["updates"] = 8                               # unequal update counts
json.dump(z, open(f"{D}/tm_twitter_aapl_s0.json", "w"))
assert subprocess.run([sys.executable, P("tm_stats.py"), D, P("ms_numbers.json")], capture_output=True, text=True).returncode != 0
z["arms"]["C"]["updates"] = 7; json.dump(z, open(f"{D}/tm_twitter_aapl_s0.json", "w"))
env = {k: v for k, v in os.environ.items() if k != "TM_ALLOW_UNFROZEN"}                                     # no frozen list, no override
assert subprocess.run([sys.executable, P("tm_stats.py"), D, P("ms_numbers.json")], capture_output=True, text=True, env=env).returncode != 0
assert S["cpu"]["status"] == "incomplete" and S["cpu"]["outcome"] == "inconclusive" and S["cpu"]["seeds_missing"] == [9]
assert S["machine_temp"]["status"] == "incomplete" and S["machine_temp"]["seeds_invalid_receiver"] == [4]
assert S["machine_temp"]["outcome"] == "inconclusive" and S["machine_temp"]["arms"]["C"]["label"] == "incomplete"
assert S["occupancy"]["outcome"] == "inconclusive" and S["occupancy"]["p_d"] == 1.0
assert S["rds_cpu_util"]["p_d"] == 4 / 1024 and S["rds_cpu_util"]["n_zero_differences"] == 1 and S["rds_cpu_util"]["outcome"] == "C higher"
assert S["cpu"]["p_d"] is None and S["cpu"]["d_ci95_bootstrap"] is not None and S["machine_temp"]["p_d"] is None
os.remove("tm_summary.json")
for line, ok in (("y  ./tm_run.py\n", False), ("x  frozen_benchmark.py\n", False), ("x *tm_run.py\n", True)):
    open("TM_FROZEN.sha256", "w").write(line)                                 # code hashes against the frozen list
    rc = subprocess.run([sys.executable, P("tm_stats.py"), D, P("ms_numbers.json")], capture_output=True, text=True).returncode
    assert (rc == 0) == ok, (line, rc)
os.remove("TM_FROZEN.sha256"); os.remove("tm_summary.json")
print("5  statistics: outcomes, exact p with ties and zeros, bootstrap interval, start-state check, incomplete and invalid handling: OK")

# 6. driver: completed attempts counted, interrupted attempts not counted, done seeds skipped
D = os.path.abspath("synth_driver"); shutil.rmtree(D, ignore_errors=True); os.makedirs(D + "/tm_runs")
shutil.copy(P("tm_driver.sh"), D)
open(D + "/tm_run.py", "w").write(
    "import sys, json, os\n"
    "if sys.argv[1] == '--check': sys.exit(0 if not os.path.exists('BROKEN') else 1)\n"
    "ds, sd = sys.argv[1], int(sys.argv[2])\n"
    "if sd == 0: json.dump({}, open(f'tm_runs/tm_{ds}_s{sd}.json', 'w'))\n"
    "else: sys.exit(1)\n")
open(D + "/BROKEN", "w").close()                                             # setup error: no attempt may start
assert subprocess.run(["bash", "tm_driver.sh", "fake"], cwd=D).returncode == 1
assert not any(f.endswith(".attempts") for f in os.listdir(D + "/tm_runs")) and "SETUP CHECK FAILED" in open(D + "/tm_runs/log.txt").read()
os.remove(D + "/BROKEN")
import fcntl
holder = open(D + "/tm_runs/tm_fake_s4.flock", "w"); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)   # a worker still running
open(D + "/tm_runs/tm_fake_s2.attempts", "w").write("1"); open(D + "/tm_runs/tm_fake_s2.running", "w").close()
open(D + "/tm_runs/tm_fake_s3.attempts", "w").write("")                       # empty marker (killed mid-write)
subprocess.run(["bash", "tm_driver.sh", "fake"], cwd=D, check=True)
att = lambda sd: open(f"{D}/tm_runs/tm_fake_s{sd}.attempts").read().strip()
assert os.path.exists(f"{D}/tm_runs/tm_fake_s0.json") and att(0) == "1"
assert att(1) == "2" and att(2) == "2" and att(3) == "2" and not os.path.exists(f"{D}/tm_runs/tm_fake_s4.attempts")
holder.close()                                                                # the worker ends: lock released
L = open(D + "/tm_runs/log.txt").read()
assert "INTERRUPTED fake s2 (not counted)" in L and "MISSING fake s1 after 2 completed attempts" in L and "rc=1" in L
assert "LOCKED fake s4 (its worker is still running; skipped)" in L
assert not any(f.endswith(".running") for f in os.listdir(D + "/tm_runs"))
subprocess.run(["bash", "tm_driver.sh", "fake"], cwd=D, check=True)           # restart: nothing re-run
assert att(1) == "2" and open(D + "/tm_runs/log.txt").read().count("start fake s1") == 2
assert att(4) == "2"                                                          # lock released: s4 now runs
print("6  driver: attempts, interruption, restart: OK")
os.chdir(HERE); shutil.rmtree("tm_synth_check")
print("ALL TM SYNTHETIC CHECKS PASSED")
