# Protocol: training on actual versus reconstructed history (reviewer point 6)

Version 3, 9 October 2026. Draft 1 (sent to the co-author on 9 October 2026) fixed the design;
this draft adds the implementation, the rules that draft 1 left open and the corrections the co-author asked
for on 9 October 2026 (the starting state of the two arms; the assumption of the test, intervals, zeros and
ties, paired failures and the scope of the conclusion), after the code was written and checked on generated
data and an independent code review. Differences from draft 1 are listed at the end. The protocol is frozen
when this file, renamed TM_PROTOCOL.md, and the SHA-256 of every file in section 8 are sent in a dated
message from the first author (O. Alani) to the co-author (M. M. Jaber). No arm of the experiment (B+ or C)
is run on any recording before that message. After freezing, nothing in this file or the code changes before
the results are reported; a necessary change is made only as a dated amendment in a separate file, stating
the reason and the results affected. Every result is reported as it comes out.

## 1. Question

The learned predictors are trained on actual sensor history but, under dual prediction, forecast from the
reconstructed history shared with the sink (suppressed readings replaced by forecasts). Does training on
reconstructed history, with the optimisation budget held equal, change their suppression or the energy
comparison with the low-cost reference? No direction is assumed. Disclosure: during development, an
exploratory run on an earlier Intel matrix (5 seeds, 20 and 40 epochs) found a difference of +0.031 with a
standard deviation of 0.064 over seeds once the budget was matched, not distinguishable from zero; it is
summarised in the docstring of the package's dual_prediction_fix.py, without per-seed outputs, and the
authors knew it when designing this study.

## 2. Subset, fixed before any run and not chosen from results

- Streams: all eleven of the main study (Intel time-indexed and the ten NAB streams), so that no stream is
  selected.
- Model: the lightweight network only (208 MACs). The CNN-LSTM is excluded for compute (about four times the
  training cost per seed); this is stated as a limitation.
- Seeds: 0 to 9. tau = 0.10 (degC for Intel, training-span standard deviations for NAB).

## 3. Arms

Phase 1 is the main-study training of seed s, exactly as run_learned.py runs it: the same training windows
(at most 4,000 sampled without replacement from the training span with numpy default_rng(0); the seven
streams with fewer windows (traffic, cpu, grok_asg_anomaly, rds_cpu_util, iio_networkin, exchange_cpc,
occupancy) use all of them, in time order), the same standardisation (training-window statistics), the same
model (frozen_benchmark.build(seed)), and 8 fits of 25 epochs with batch 64 and Keras's internal 15%
validation split (the last 15% of the window arrays, which are not trained on and drive Keras's loss curve
only), each followed by scoring on validation suppression. Its validation-selected checkpoint reproduces the
main-study entry of that seed; this is recorded for every seed (pre-freeze check: reproduced bit for bit for
traffic seed 0, intel_time seed 0, cpu seed 7 and exchange_cpc seed 13, and again for traffic seed 0 with
the final code, whose phase 1 differs only by the added state hash). A seed whose phase 1 does not reproduce
stays valid, because both arms share its starting point, and the number of such seeds is reported. Phase 1
also scores the test span, with the receiver check, to check this reproduction and to give the context label
of section 4; those numbers are already published.

Starting state. Both arms branch independently from the same epoch-200 weights and the complete optimiser
state. The state is captured immediately after the last phase-1 fit, before any validation-selected weights
are restored, and saved with its Adam state (keras .keras file); the selected weights are loaded into the
phase-1 model only after that, to score the test span, and the epoch-200 weights are then put back for the
closed-loop pass of arm C. Each arm loads the saved file separately. Before each arm trains, its weights,
step count and every optimiser variable are checked equal to the captured state, and the SHA-256 of each
arm's starting state is recorded; the two must be equal. Both arms use the same sampled window positions in
the same order, and so the same Keras holdout (the last 15% of the window array, by position) and the same
targets; only the inputs of arm C differ. Both are reseeded with seed + 1000. In the Keras version used, the
holdout is taken by position before shuffling, and the shuffle is a permutation of the window indices that
depends only on the seed and the number of training windows, not on their values; once the seed is set, the
same permutation is used in every epoch of the phase, as in the main study (seed s in phase 1). The two arms
therefore see the same batch order, and they train for the same number of epochs on the same number of
windows; the number of optimiser updates of each arm is recorded and must be equal. A run that fails any of
these checks stops without a result. Pre-freeze checks on generated data: with identical windows the two
arms are identical (same validation history, checkpoint and test suppression), which also confirms the batch
order; the captured state equals that of a fresh phase-1 training without validation scoring, including a
seed whose phase 1 selected a checkpoint before the cap and whose two arms are then run, so it is the
epoch-cap state and not a restored checkpoint; the check fails, as it should, when the code is altered to
restore the selected weights before the capture or to save them in the file.

The two arms:
- B+ (actual history, matched budget): 200 further epochs on the same actual-history windows.
- C (reconstructed history): the epoch-200 model is run in closed loop over the training span (tau, shared
  state, the first 12 readings synchronised, exactly the loop of the package evaluator); the training
  windows are rebuilt from that reconstructed series at the same positions and with the same sampled indices
  (inputs reconstructed, targets the actual readings, the standardisation of phase 1); 200 further epochs on
  them. In both arms a checkpoint is taken every 25 epochs of the second phase (epochs 225 to 400) and the
  one with the highest validation suppression is kept (ties: the earlier). The test span is then scored
  once, with the independent-receiver check of the package (sender and receiver reconstructions identical
  and the error bound held at the receiver); a failed check is recorded, not raised.

Reconstruction cost, reported: the closed-loop pass costs one forward pass per training step and node (208
MACs each): from 551 (traffic) to 12,551 x 30 (Intel) passes; a one-off training cost, not a per-round
deployment cost.

## 4. Analysis (tm_stats.py)

Per stream, paired over seeds: d = test suppression(C) - test suppression(B+). Estimand: mean d over seeds.
Test: two-sided sign-flip permutation test of the mean (all 2^10 sign assignments), computed on the integer
differences in transmitted readings (both arms score the same test steps). Assumption: under the null
hypothesis the paired seed differences are independent and symmetric about zero (sign-exchangeable); the
enumeration makes the p-value exact under that assumption, not without it. Zeros and ties: a seed with a
zero difference contributes nothing to any sign assignment and is kept in the count of seeds; assignments
whose absolute sum equals the observed one count as at least as extreme, which integer arithmetic makes
exact; if every difference is zero, p = 1. Family: 11 streams, Bonferroni 0.05/11. Outcome per stream: C
higher, B+ higher, or inconclusive. Interval: the 95% percentile bootstrap interval of the mean d over seeds
(10,000 resamples of the seeds with replacement, numpy default_rng(0) started afresh for each stream,
numpy's default linear percentile), reported for every stream as a description; with 10 seeds it is
approximate and does not decide the outcome. With 10 seeds the smallest attainable p is 2/1024 and
significance needs at least 9 seeds of one sign, so power is limited.

Scope of the conclusion. The result is stated only for one round of reconstructed-history retraining (200
further epochs from the epoch-200 main-study state) of the lightweight model, on these eleven recordings, at
tau = 0.10, and describes training randomness on fixed recordings. An inconclusive outcome is reported as
inconclusive: it is not evidence that training mismatch has no effect, and the paper will not say so.

Descriptive: each arm's net saving per node-round against the validation-selected low-cost reference of the
main study (ms_numbers.json: selected arm, its test suppression and MACs), g = e_r*(rho - rho_ref) - (208 -
M_ref)*E_op at e_r = 3.8e-4 J and 1 nJ per MAC, its mean, and a label from the same test at the same
threshold and under the same assumption (saves, costs, ns), computed in exact integer units; also for the
phase-1 checkpoints of the same 10 seeds, as context, if their receiver checks passed. These labels are not
the main-study verdicts (Wilcoxon over 20 seeds, 0.05/22). tm_stats.py requires every run to carry the same
code hashes, software versions and settings (section 7), the same input-data hash within a stream, equal
start states and update counts in the two arms, and code hashes listed in TM_FROZEN.sha256 and equal to it
(it stops if that file is missing; only the synthetic check may override this, with TM_ALLOW_UNFROZEN=1).
Whether the label differs between B+ and C is reported.

## 5. Failures

- Before any attempt the driver checks the flock command, the software versions (section 7) and that every
  listed stream loads; a setup error stops it without counting an attempt.
- One process per (stream, seed) (tm_driver.sh). A seed gets at most two completed attempts; an attempt is
  completed when the process returns, whatever its exit status, and is logged with that status. An attempt
  whose return the driver did not see (after an interruption of the machine or of the driver) is logged as
  interrupted and not counted (runs are deterministic, so a repeat cannot select on outcomes). Each seed is
  guarded by a kernel lock that its worker inherits, so a seed whose worker is still running is skipped;
  drivers run on disjoint stream lists.
- A seed without a result after two completed attempts is missing; a seed whose receiver check fails in
  either arm is invalid. Failures are handled in pairs: a missing or invalid seed is removed from both arms,
  never from one. A stream with fewer than 10 valid seeds is incomplete: its mean d and interval over the
  valid pairs are reported as descriptive only, with no p-value; its outcome is inconclusive, its energy
  labels read "incomplete", and the family is not reduced. The number of missing and invalid seeds is
  reported for every stream.
- No model is retrained, retuned or re-seeded after test results have been seen, except a bug-fix rerun
  (same seeds and settings, invalidated outputs kept), declared in an amendment.

## 6. Compute

Phase 1 takes about 35 s (traffic) to 65-85 s (Intel) per seed on one core; a full run is about three times
phase 1. 110 runs: about 5-8 hours on one process, about half with two drivers on disjoint stream lists.

## 7. Commands (run directory set up as in the package README, plus the files of section 8)

    sha256sum -c TM_FROZEN.sha256
    PYTHONDONTWRITEBYTECODE=1 python3 tm_synthetic_check.py   # generated data only, in a subfolder
    bash tm_driver.sh intel_time taxi traffic cpu machine_temp &
    bash tm_driver.sh grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy; wait
    python3 tm_stats.py                           # -> tm_summary.json

Software: Python 3.11 with the package's requirements.txt (numpy 2.4.4, tensorflow 2.21.0, keras 3.15.1);
one thread per process (set by tm_run.py). Each run records the versions, the settings, the SHA-256 of the
code it imports and of its input file.

## 8. Frozen files (SHA-256 listed in TM_FROZEN.sha256 at freezing)

tm_run.py, tm_stats.py, tm_driver.sh, tm_synthetic_check.py; from the package (unchanged):
frozen_benchmark.py, run_learned.py, fastfwd.py, dual_prediction_fix.py, nab_data.py, requirements.txt;
inputs: intel_time.npy and the ten NAB csv files at the paths nab_data.py reads (nab_repo/data/...);
main-study results: ms_numbers.json and learned_<stream>.json.

## Differences from draft 1

- Implementation stated: phase 1 reproduces the main-study training; the epoch-200 model and Adam state are
  saved and reloaded; both arms reseeded with seed + 1000.
- Co-author, 9 October 2026, point 1: the starting state is defined (captured before any validation-selected
  weights are restored; same window positions, holdout membership, batch order and update counts), and the
  equality of the two starting states is tested and recorded in every run (state SHA-256, update counts),
  with a pre-freeze check that the captured state is the epoch-cap state.
- Co-author, 9 October 2026, point 2: the test's assumption (sign-exchangeability) is stated; a bootstrap
  interval of the mean difference, the handling of zeros and ties and paired failure handling are added; the
  conclusion is limited to one round of reconstructed-history retraining of the lightweight model, and an
  inconclusive outcome is not reported as no effect.
- "The same 4,000 sampled indices" corrected: seven streams have fewer windows and use all of them.
- Analysis: the exact test is computed on integer transmission counts (ties); the descriptive energy label
  is defined (it was open in draft 1) and the phase-1 context added; power stated; runs must share code,
  versions and settings, the input-data hash within a stream, and match TM_FROZEN.sha256.
- Failures: a setup check (flock, versions, data) before any attempt; only completed attempts count;
  interrupted attempts (machine or driver) are logged and not counted; a kernel lock per seed; a failed
  receiver check (either part of the package check) is recorded as invalid rather than ending the process.
- Compute estimate and the pre-freeze checks added.
- New: sections 7 (commands and software) and 8 (frozen files), and the freezing and amendment procedure in
  the preamble; "energy verdict" in section 1 became "energy comparison", with the label defined in section
  4.
- Details added: ties between checkpoints go to the earlier one; the first 12 readings of the closed-loop
  pass are synchronised; the one-off reconstruction cost is reported.
- The earlier exploratory run is now described with its source (the docstring of dual_prediction_fix.py, a
  summary without per-seed outputs) and stated as known to the authors when the study was designed.
