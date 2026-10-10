# Matched comparison with published dual-prediction schemes (Table 10)

This document holds the settings and checks behind Table 10 and the Section 6.8 paragraph of
the paper. References in square brackets are those of the paper: [10] Santini and Römer (LMS
dual prediction), [25] Fathalla et al. (LSTM-DPS), [24] Fathy et al. (AM-DR).

## 1. Sources and ports

LMS dual prediction [10] and AM-DR [24] are ported line by line from the MATLAB code released
with [24] (github.com/YasminFathy/AMDRIoT: `baseline/dual_prediction.m`, which is that code's
implementation of [10], and `myapproach_AMDR/AMDR_dual_prediction.m`) in `src/ports_faithful.py`.
LSTM-DPS [25] is ported from the Python code released with [25]
(github.com/Ahmed-Fathalla/LSTM-DPS-for-IoT) in `src/lstmdps_port.py` (transmission loop and
NumPy forward pass) and `src/lstmdps_train.py` (training).

The repositories were used at these commits: AMDRIoT `bf1a67b89535be65ce30cc65cad1932258cb875d`
(5 January 2018) and LSTM-DPS-for-IoT `2a4347e08c94a1d074ad82d7b3b51b6f16c2ce19` (4 December 2023).
The released Intel data file of LSTM-DPS, `data/Intel-DS.csv` extracted from `data/Intel-DS.rar`, has
SHA-256 `a48ef6f172006990e51dae393f0426d67e55597c057238b653f066eba102702f`.

For the protocol of the paper, LMS and AM-DR are re-expressed as stateful predictors in
`src/matched_baselines.py`. With the protocol's synchronised 12-row warm-up, the re-expressed
schemes make the same transmission decisions as the line-by-line ports on all 30 Intel motes:
0 disagreements in 226,170 test rounds for each of four settings (published and
validation-selected, LMS and AM-DR, released rule; `results/port_validation.json`, key
`harness_equivalence`).

## 2. Reproduction of published results

**LMS and AM-DR.** On the authors' data selection (the readings of each mote dated 6–9 March
2004, in file order, as in `dataset/load_data.m` of [24]) and e_max = 0.5 °C, the ports give the
transmitted shares in Table S1. Four of the five values are within 0.8 percentage points of the
published ones. The fifth, Fig. 8 of [24], shows about 7% for mote 11, where the AM-DR port gives
3.5% with the parameters printed in that figure.

**Table S1.** Transmitted share of readings: values reported in [24] and values from the ports.

| Scheme | Mote | Parameters (as stated in [24]) | Reported in [24] | Port |
|---|---|---|---|---|
| LMS [10] | 11 | N = 5, µ = 10⁻⁵ (default settings, Figs. 3–4) | 92% reduction (8% transmitted) | 7.4% |
| AM-DR | 11 | w_f = 5, w_s = 10, α = 10⁻⁷ (default settings, Figs. 3–4) | 95% reduction (5% transmitted) | 4.3% |
| LMS [10] | 30 | N = 4, µ = 10⁻⁵ (Fig. 9) | 18% transmitted | 17.8% |
| AM-DR | 30 | w_f = 4, w_s = 8, α = 10⁻⁷ (Fig. 9) | about 13% transmitted | 13.4% |
| AM-DR | 11 | w_f = 4, w_s = 8, α = 1.2×10⁻⁷ (Fig. 8) | about 7% transmitted | 3.5% (not reproduced) |

With the published α, AM-DR transmits on exactly the same rounds as with α = 0 for all five
motes of that selection (1, 11, 13, 30 and 49): at that α, AM-DR is persistence-like prediction
from the shared means plus the released transmission rule.

**LSTM-DPS.** Driven by the released trained models (the `_refit.h5` files) and the released
data file, the port reproduces the total transmission count exactly in all 24 released
experiments on motes 1 and 11 (12 tolerances from 0.1 to 4.0 °C each;
`results/lstmdps_port_validation.csv`). This checks the transmission loop and the forward pass;
it does not check retraining. On the same series at 0.10 °C, LSTM-DPS suppresses 94.0% (mote 1)
and 95.0% (mote 11) and persistence 92.7% and 94.2% (column `persistence_pct`; persistence
transmits the first reading and then every reading that differs from the last one sent by more
than the tolerance). The other three motes cannot be rerun as released:

- the released data file contains no readings of motes 30 and 49: it is ordered by mote and
  stops at 1,048,575 readings, the row limit of a spreadsheet, partway through mote 27;
- it contains 6,428 readings of mote 13, where 11 of its 12 saved runs used 6,132. The
  twelfth file (`Exp Intel-DS Mote_13 Epsilon_0.3.pkl`) holds a run with 8,337 readings and
  mote ID 1 in its own summary. The CSV lists the replayed counts of mote 13 for the record.

## 3. Settings under the protocol of the paper

**State.** The schemes start from zero at the beginning of every slice (w[0] = 0 for LMS and
λ[0] = 0 for AM-DR, as in both papers), and the 12 warm-up rows are processed as transmitted
rounds. Under the released rule the error window starts with unknown errors, so the first
W − 1 test rounds transmit.

**Two transmission rules.** The released rule suppresses a reading only if the last N (LMS) or
w_f (AM-DR) errors are all below τ. The single-error rule of Section 3.3 suppresses when the
current error is at most τ.

**LMS.** N ∈ {4, 5, 6, 8, 10}, the range given in [10]. The step size is µ = c / E_x, where E_x is
the mean square of the training span and c ∈ {10⁻⁶, 3×10⁻⁶, …, 0.3, 1} (13 values; c = 1 is the
convergence bound of Eq. (5) in [10]). On Intel, the published µ = 10⁻⁵ corresponds to
c ≈ 4.5×10⁻³.

**AM-DR.** (w_f, w_s) ∈ {(4, 8), (5, 10)}, the two published pairs. The mixing rate is
α = c / σ², where σ² is the training-span variance and c ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 0.1, 1, 10, 100}.
On Intel, the published α = 10⁻⁷ corresponds to c ≈ 1.1×10⁻⁶.

**Selection.** Parameters are selected on validation suppression, ties going to the first grid
entry, and test is scored once with the selected parameters.

**Grid extensions.** Two earlier runs used narrower grids of c. Every extension was triggered
by a validation selection at a grid edge, not by test results; the earlier runs are kept in
`results/matched_earlier_grids/` (see its README). In the final grids, the remaining edge
selections are ties at zero validation suppression, which go to the first grid entry, or α = 0.

**Table S2.** Parameters selected on validation for each stream, with µ = c / E_x for LMS and
α = c / σ² for AM-DR. † Validation suppression is zero at every grid point, so the tie goes to the
first entry.

| Stream | LMS, released rule: N, c | LMS, single-error: N, c | AM-DR, released rule: (w_f, w_s), c | AM-DR, single-error: (w_f, w_s), c |
|---|---|---|---|---|
| Intel temperature | 4, 0.01 | 4, 0.1 | (4, 8), 0.01 | (5, 10), 1 |
| taxi | 4, 0.03 | 6, 0.1 | (4, 8), 0.01 | (4, 8), 0.1 |
| traffic | 4, 1e-6 † | 10, 0.01 | (4, 8), 0 † | (4, 8), 0.001 |
| cpu | 4, 1e-6 † | 5, 0.03 | (4, 8), 0 † | (5, 10), 0.1 |
| machine_temp | 4, 0.01 | 4, 0.03 | (4, 8), 10 | (5, 10), 10 |
| grok_asg_anomaly | 4, 0.01 | 10, 0.1 | (4, 8), 0 | (4, 8), 0 |
| rds_cpu_util | 4, 0.01 | 4, 0.03 | (4, 8), 0.1 | (5, 10), 0.1 |
| iio_networkin | 4, 0.03 | 6, 0.03 | (4, 8), 1 | (5, 10), 10 |
| twitter_aapl | 4, 0.03 | 6, 0.0001 | (4, 8), 0.01 | (4, 8), 0.01 |
| exchange_cpc | 4, 1e-6 † | 5, 0.1 | (4, 8), 0.01 | (4, 8), 0.1 |
| occupancy | 4, 0.003 | 6, 0.03 | (4, 8), 0.01 | (5, 10), 0.01 |

With the released rule, validation selects N = 4 and (w_f, w_s) = (4, 8) on every stream: the
shortest windows, and so the least strict form of that rule. The largest sink error over the 44
LMS and AM-DR test runs is 0.99999 τ.

**LSTM-DPS.** The released sequence runs on the test slice unchanged: a 3-reading
initialisation, a 20-step LMS phase whose weights stay fixed (the released loop never adapts
them), a refit of 2 epochs on the resulting aggregate list, and the LSTM phase. The model is
trained per mote on the last 24 h of the training span (2,597 readings, 3 March 00:11 to
4 March 00:10). Training follows the released procedure, re-implemented in TensorFlow 2: a TPE
search with five trials over the released search space (10 epochs, MSE loss), then two fits of
the chosen architecture (MAE loss, best validation MAE kept). This is the setting of 54 of the 60
saved Intel experiments; the other six used one trial and one fit. The Keras 2 hard-sigmoid
recurrent activation of the released models (clip(0.2z + 0.5, 0, 1)) is passed explicitly,
because Keras 3 redefines `hard_sigmoid`. The optimiser state at the refit is that of the last
training epoch, not of the checkpoint. The NumPy forward pass used for scoring is asserted equal
to Keras for every deployed model. The update mechanism is disabled, as in all 60 saved Intel
experiments, and each mote is run five times (seeds 0–4), as in the released run script. The
transmission rule is the released one, |x − x̂| < τ suppresses, which is a single-error rule.

## 4. Compute charge per round

- **LMS:** 2N + 1 MACs, the per-iteration multiplication count of [10], charged in every round
  as an upper bound, since the weight update runs only on transmitted rounds.
- **AM-DR:** w_f + 4 MACs: w_f − 1 for the fast mean, 1 for the slow running mean, 2 for the
  convex mix and 2 for the λ update.
- **LSTM-DPS:** the MAC count of the selected architecture, 4u(u + 1) per time step over 3 steps
  for an LSTM of u cells, plus its dense layers; across the 150 runs this ranges from 900 to
  3,428 MACs.

Net saving against the reference, per node-round:
g = e_r · (ρ − ρ_ref) − (MACs − MACs_ref) · E_op, with e_r = 3.8×10⁻⁴ J and E_op = 1 nJ.

## 5. Files

| File | Role |
|---|---|
| `src/ports_faithful.py` | Line-by-line ports of the two MATLAB scripts |
| `src/matched_baselines.py` | LMS and AM-DR under the protocol: selection on validation, one test score, independent receiver |
| `src/lstmdps_port.py` | LSTM-DPS transmission loop and NumPy forward pass |
| `src/lstmdps_train.py` | LSTM-DPS training (hyperopt TPE, two fits, refit) |
| `src/lstmdps_protocol.py` | LSTM-DPS on the Intel test slice, sender and independent receiver |
| `src/validate_ports.py` | Table S1, the α = 0 identity, the harness equivalence, the LSTM-DPS replay |
| `src/summarise_matched.py` | Table 10 and the counts of Section 6.8 |
| `results/matched_<stream>.json` | One per stream: the four selections, their validation grids and test scores |
| `results/lstmdps_intel_runs.jsonl` | The 150 LSTM-DPS runs (30 motes × seeds 0–4) |
| `results/matched_summary.json` | Per stream and rule: ρ, Δρ against the reference, MACs, g; the derived counts (`_derived`); the published settings at 0.50 °C on the Intel test span |
| `results/lstmdps_summary.json` | LSTM-DPS against persistence on the Intel test span |
| `results/port_validation.json`, `results/lstmdps_port_validation.csv` | The checks of Sections 1 and 2 |
| `results/matched_earlier_grids/` | The two earlier runs with narrower grids |

## 6. Rerunning

Inside the `run/` directory set up as in the README (the `cp src/*.py results/*.json …` step
also copies the files of this comparison; copy `results/lstmdps_intel_runs.jsonl` as well):

```bash
for s in intel_time taxi traffic cpu machine_temp grok_asg_anomaly rds_cpu_util \
         iio_networkin twitter_aapl exchange_cpc occupancy; do
  python3 matched_baselines.py $s                  # -> matched_<s>.json
done
python3 summarise_matched.py                       # -> matched_summary.json, lstmdps_summary.json
python3 validate_ports.py --lstmdps PATH           # -> port_validation.json, lstmdps_port_validation.csv
```

`validate_ports.py` needs `raw/data.txt` (as for `text_checks.py`) and, for the LSTM-DPS
replay, a clone of github.com/Ahmed-Fathalla/LSTM-DPS-for-IoT with `data/Intel-DS.rar`
extracted to `data/Intel-DS.csv` (for example `unrar x data/Intel-DS.rar data/`); without
`--lstmdps` it skips the replay. The replay reads the released experiment files, which are
pickles of the authors' classes: loading them imports the authors' `utils` package and
hyperopt, so it needs h5py, matplotlib and hyperopt (all in `requirements.txt`) but not
TensorFlow.

LSTM-DPS (about 5.4 CPU-hours for the 150 runs, single-threaded):

```bash
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1
python3 lstmdps_protocol.py 0,1,2,3,4 $(seq -s, 0 29) lstmdps_rerun.jsonl
```

The script appends one line per run and skips runs already in the output file, so an
interrupted run can be restarted with the same command (and, given the shipped file as output,
it would skip every run). `summarise_matched.py` reads `lstmdps_intel_runs.jsonl`; to summarise
a rerun, rename its output to that name. The shipped file was produced by
three parallel workers and then sorted by seed and mote.

At the largest grid values (LMS step size, AM-DR mixing rate) the scheme diverges on some
streams and NumPy prints overflow warnings: the LMS weights, or AM-DR's unclipped λ, grow
without bound. From that point the prediction is not a finite number and every reading is
transmitted. None of the 44 selected settings produces a non-finite prediction on its validation
or test slice, and every selected run passes the independent-receiver check.

**Determinism.** `matched_baselines.py`, `summarise_matched.py` and `validate_ports.py` are
deterministic. Across operating systems, every output except one field is identical: a rerun on
Windows 11 (Python 3.11.9, the versions in `requirements.txt`) reproduced the eleven
`matched_*.json`, both summaries and `lstmdps_port_validation.csv` exactly, and
`port_validation.json` differs only in `max_sink_err_LMS` for motes 11, 13, 30 and 49, from the
twelfth significant digit on (for example 0.4992775136235146 against 0.49927751362369577). That
value comes from the line-by-line LMS port, whose prediction uses `np.dot`; `np.dot` goes through
the platform's BLAS library, which may sum in a different order. Every transmission decision and
every other number is the same. LSTM-DPS training is seeded (`keras.utils.set_random_seed` and a seeded hyperopt
`rstate`) and was run single-threaded on CPU. With the versions in `requirements.txt`, a rerun of mote 0, seed 0
reproduced the shipped line exactly, in every field except the wall-clock `seconds`. A different
machine, TensorFlow build or thread count can change the order of floating-point operations, and
with it the architecture the search selects or the trained weights; that rerun is the only one
checked.
