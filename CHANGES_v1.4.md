# Changes in version 1.4

Version 1.4 adds four things. Part 1 is a retrospective diagnostic for the matched
comparison of version 1.2: a crossed comparison of the two transmission rules of LMS and AM-DR with the
parameters held fixed (Section 6.8). Part 2 replaces the per-MAC sensitivity of Section 6.6
with inference energy on four processors (Section 6.6; three of them in Table 6). Part 3 documents the
compute undercount and the untuned baselines quoted in the Introduction and Section 7. Part 4 (at the end) revises the presentation of
every figure for the journal's artwork requirements, adds the script of Figure 1, and replaces absolute paths
of the working directory. No existing file is changed
apart from `README.md`, `MATCHED_BASELINES.md` (reference numbers now follow the current manuscript: [13],
[18] and [25] of the earlier draft are [10], [25] and [24]), one comment line added to `requirements.txt`,
two places of `CHANGES_v1.3.md` (its status, and the sentence on the manuscript figures and their
hashes), one phrase and the reference numbers of `CHANGES_v1.2.md`, the licence note of `LICENSE` (the NAB
files are MIT-licensed, not AGPL-3.0; the notice is added as `inputs/nab/LICENSE_NAB.txt`), one
sentence of the module docstring of `src/frozen_benchmark.py` (shortened; no executable line changed, which
`ast` confirms), and the files listed in Part 4 (figure scripts and figures; path lines); Table 10,
`matched_*.json` and every saved suppression are as in version 1.3.

## Part 1: crossed rule comparison (Section 6.8)

Why. In `matched_baselines.py` the parameters of each scheme are selected on validation separately for
each rule, so the rise in suppression from the released rule to the single-error rule combines the rule
change with the reselection. Here each of the two selected parameter sets is scored on the test span
under both rules, so that only the rule changes.

Added
- `src/rule_crossed.py`: for each scheme and stream, the parameter set selected under the released rule
  and the one selected under the single-error rule (read from `matched_<stream>.json`) are each scored
  on the test span under four rule variants: the released rule as released (the last W errors all
  `< tau`, W = N for LMS and w_f for AM-DR), the released rule with the `<= tau` boundary of Section 3.3,
  the single-error rule as published (`<= tau`) and the single-error rule with `< tau`. Data, tolerance,
  warm-up and initial state are those of `matched_baselines.py`; the scheme's state evolves under its own
  transmission decisions. Every run is checked with the independent receiver of `matched_baselines.py`
  and asserts that the sink error never exceeds tau. Reports suppression, transmissions, mean and
  maximum sink error, and the number of errors exactly equal to tau.
- `src/rule_crossed_check.py`: an independent scalar re-implementation (per node, plain Python loops,
  importing neither `matched_baselines.py` nor `rule_crossed.py`) that re-scores all 176 runs and asserts
  that every transmission count is identical. Its sums are taken left to right (`lsum`): the built-in
  `sum()` of Python 3.12 and later uses compensated float summation, under which an earlier version of
  this checker failed four Intel counts (found in the authors' review); with `lsum` all 176 counts match under both
  Python 3.11 and Python 3.12.
- `src/rule_crossed_summary.py`: writes `rule_crossed_summary.csv` and prints the numbers of Section 6.8.
- `results/rule_crossed_<stream>.json` (eleven files) and `results/rule_crossed_summary.csv`
  (176 rows: 11 streams x 2 schemes x 2 parameter sets x 4 rule variants).

Checks
- The two published cells of each scheme and stream (released-rule parameters under the released rule,
  single-error parameters under the single-error rule) reproduce the suppression of `matched_*.json`
  (asserted in `rule_crossed.py`). Their transmissions and mean sink errors are also identical; the
  maximum sink error is identical except in one cell (AM-DR on grok_asg_anomaly, released rule), where
  it differs in the last digit because `matched_baselines.py` reports it from the sender side.
- No error equals tau exactly in any of the 176 runs, so the `<` and `<=` variants of each rule give
  identical transmissions; the controlled comparison therefore holds under either boundary.
- Every run passes the independent-receiver check; the largest sink error is 0.999997 tau.
- `rule_crossed_check.py` reproduces every transmission count of the 176 runs, under Python 3.11 and 3.12.
- From a clean unpack of this package on Linux, `rule_crossed.py` regenerates the eleven
  `rule_crossed_*.json` byte for byte and `rule_crossed_summary.py` regenerates the CSV byte for byte.

Result (quoted in Section 6.8)
- With the parameters selected under the single-error rule, the released rule lowers suppression on all
  22 scheme-stream pairs, by 0.06-0.43.
- With the parameters selected under the released rule, the single-error rule raises suppression on all
  19 pairs in which either rule suppresses anything, by 0.02-0.38. On the other three (LMS on traffic,
  cpu and exchange_cpc) the released-rule parameters transmit every reading under both rules; on all
  three, every grid setting had zero validation suppression under the released rule, so the first grid
  entry was selected. AM-DR on traffic and cpu was also selected by default in this way (first grid
  entry: w_f = 4, w_s = 8, c = 0); those two pairs are among the 19, because the single-error rule suppresses with them
  (by 0.06 and 0.19).

Run order for the added files (continuing the numbering of `CHANGES_v1.2.md`; inside `run/`, after step 16).
The run-directory setup of the README copies the shipped `rule_crossed_*.json` into `run/`; step 20
overwrites them, so run it before steps 21 and 22 to check fresh outputs rather than the shipped ones.

20. `python3 rule_crossed.py <stream>`        for each of the eleven streams -> `rule_crossed_<stream>.json`
21. `python3 rule_crossed_check.py <stream>`  for each of the eleven streams (asserts; writes nothing)
22. `python3 rule_crossed_summary.py`         -> `rule_crossed_summary.csv`

## Part 2: inference energy on processors (Section 6.6, Table 6)

Version 1.4 also replaces the per-MAC sensitivity of Section 6.6 (0.5, 1 and 2 nJ per MAC) with
inference energy on four processors (three of them in Table 6; STM32H7 only in the text). No learned model is retrained and no suppression changes; the
per-seed test suppression of `learned_<stream>.json` and the reference of `ms_numbers.json` are reused.

What each number is
- The verdicts on GAP-8, STM32L4 and STM32H7 combine the floating-point suppression of `learned_<stream>.json`
  with per-MAC energies estimated from other workloads; the energy of these models on those processors is
  not measured, and their INT-8 suppression (assumed on STM32L4 and STM32H7) is not tested.
- Reported (STM32L4, STM32H7): the efficiencies of an INT-8 CIFAR-10 network in PULP-NN (Garofalo et al.,
  Phil. Trans. R. Soc. A 378: 20190155, 2020; reference [26] of the paper): GAP-8 16.1-24 GMAC/s/W (used
  here only through the two ratios), STM32L4 (Cortex-M4) 1.70 GMAC/s/W (16.1/9.48 and 24/14.1 agree),
  STM32H7 (Cortex-M7) 0.97 or 0.61 GMAC/s/W (the paper's two ratios disagree, so both are used).
- Derived (STM32L4, STM32H7): E_op = 1/efficiency (STM32L4 0.59, STM32H7 1.03-1.64 nJ per MAC);
  overhead counts from the architectures (CNN-LSTM: 1,152 elementwise products, 1,920 sigmoid/tanh,
  2,305 bias adds and ReLUs per inference; lightweight: 33 bias adds and ReLUs); parameter counts
  (225 and 8,481); model-transfer energy = parameters x bits x 50 nJ/bit (Section 3.1), received once.
- GAP-8 (`gap8_lstm_energy.py`). Reported in Cappellone et al., IEEE DFT 2020, doi
  10.1109/DFT50435.2020.9250796 (reference [27] of the paper): the parameter counts of a Seq2Seq LSTM
  forecaster (Table I), its execution cycles per prediction on the GAP-8 cluster in 16-bit fixed point
  (Q3.12) with weights in L2 (Table IV), a cluster clock of 175 MHz and 75 mW quoted as worst-case
  consumption (not a measurement). Derived: the architecture reproduces Table I exactly; MACs per prediction
  = 36 x 4 x (n + n^2) + 9n, so 1.434, 1.441 and 1.615 MACs per cycle for 60, 100 and 150 units.
  Timing: the "previsions/s" column of Table IV equals 9 outputs per prediction at 150 MHz in all ten rows
  (LSTM and GRU), not at 175 MHz; cycles, and so MACs per cycle, do not depend on the clock, but the energy
  does through the assumed power, so both clocks are used. Assumed power 65 mW (0.27 mJ per inference at
  241 inferences per second, the PULP-NN measurement on GAP-8, made at 170 MHz and applied unchanged at
  175 MHz, which makes the lower bound of 0.230 nJ about 3% optimistic; no verdict depends on it) or 75 mW,
  with no reduction at 150 MHz.
  E_op = power / (clock x MACs per cycle): 0.230-0.299 nJ per MAC at 175 MHz and 0.268-0.349 at 150 MHz;
  the four values 0.230, 0.268, 0.299 and 0.349 are used (`EOPS`).
- Assumed (GAP-8): implementation transfer, that our CNN-LSTM runs at the source's MACs per cycle. The source
  uses hard-sigmoid, its own tanh approximation and 32-bit biases; our 16-bit implementation (`src/m0/`)
  uses interpolated tables, and the fixed-point check below covers ours, not theirs. The source's cycles
  already contain its activation, memory and control work, so only MACs are charged (no MAC-equivalent
  overhead, which would count that work twice); charging the overheads of `platform_energy.py` as well
  changes no verdict, while charging ten times those overheads on top would remove the taxi saving at
  0.299 and 0.349 nJ (not significant and a loss). Weights are sent once as 16-bit.
- Rate margin (GAP-8): our CNN-LSTM has 32 LSTM units, fewer than the 60-150 units whose rate is used, and
  smaller layers may run at a lower MAC per cycle. With E_op = power / (clock x MACs per cycle), the taxi
  saving stays significant down to 1.17 MACs per cycle in the worst scenario (75 mW at 150 MHz; 1.24 with
  overheads), 18% (14%) below the slowest rate of the source (1.434, 60 units), and down to 0.87-1.07 in the
  other three scenarios (25-39% below); `taxi_min_mac_per_cycle` in `gap8_lstm_energy.json`.
- Power basis (GAP-8): the 65 mW is the 1.2 V operating point of PULP-NN (0.27 mJ per inference at 241 per
  second, 170 MHz). Its 1 V point (0.19 mJ at 127 per second, about 90 MHz from the frame rates) costs
  0.70 times the energy per cycle (0.269 against 0.383 nJ), which at the source's 1.43-1.62 MACs per cycle
  gives 0.17-0.19 nJ per MAC (0.269 divided by each). There the CNN-LSTM on exchange_cpc becomes not
  significant instead of a significant loss: computed with `stream_g`, it saves significantly only below
  0.132 nJ (0.125 with overheads) and loses significantly above about 0.207 nJ (0.196 with overheads).
  Taxi and every other verdict are unchanged, with and without overheads, so it still saves only on taxi.
- Simulated: Cortex-M0 energy per inference. The five predictors were written in C (`src/m0/`), in a
  16-bit fixed-point form (Q12 weights and activations, int32 accumulators, sigmoid/tanh from 513-point
  interpolated tables, per-stream integer scaling of the readings) and in float32; compiled with
  arm-none-eabi-gcc 14.2.1 (xPack 14.2.1-1.1) for -mcpu=cortex-m0 at -O2, -O3 and -O3 -funroll-loops;
  run in the instruction-set simulator of Nikov et al. (reference [21]; github.com/PicoPET/
  thumbulator-stm32f0x, tag teamplay-D4.5, commit c55ea1b, unmodified and built with -fcommon; a second copy
  with program printing enabled is used only to read the predictions back); the event counters
  between two GPIO markers around one prediction were converted to energy with the ten hardware-fitted
  models of Table II of [21]. Counts are averaged over 16 Intel test windows. The Cortex-M0 energies of
  all eleven streams use the intel_time seed-0 weights and 16 windows of the first included mote (column 0)
  of the Intel test span; exponential smoothing is built with alpha = 0.3 and linear extrapolation with
  K = 2 (the K selected on taxi, the only stream whose reference is linear extrapolation). At -O2 the smoothing
  weights do not change the instruction count of the fixed-point build; at -O3 and -O3 -funroll-loops the compiler folds these constant
  weights, so the count of the smoothing reference depends on alpha (34-117 instructions, excluding MULS, over the
  alphas selected on the nine smoothing streams, against 76-80 at alpha = 0.3). Building each stream with its own
  alpha changes no verdict; in Table 6 it changes one printed value: the upper Cortex-M0 bound of the CNN-LSTM on
  traffic moves from 0.2149999 to 0.2150139, which would round to 0.22 instead of 0.21. The float32 builds use their own standardisation constants and serve
  only as a cross-check; the reported energies are those of the fixed-point builds.
- Assumed: overhead cost in MAC-equivalents on STM32L4 and STM32H7 (elementwise product 1,
  sigmoid/tanh 2, bias add or ReLU 0.25; a tenfold activation charge changes no verdict there); weights
  sent once as 8-bit (STM32L4, STM32H7) or 16-bit (GAP-8, Cortex-M0), amortised over 10^5 rounds (no
  verdict changes at 10^4 rounds on any processor).
- Revised before release: the draft of version 1.4 circulated among the authors took GAP-8 from PULP-NN as well
  (0.042-0.062 nJ per MAC, an INT-8 convolutional workload), under which the CNN-LSTM also saved on
  exchange_cpc and was not significant on traffic. That figure was replaced by the 16-bit LSTM figure
  above, which is closer to the workload; on exchange_cpc the CNN-LSTM saves significantly only below
  about 0.132 nJ per MAC.

Added
- `src/m0/`: `build_sim.sh` (fetches and builds the simulator at teamplay-D4.5 and the toolchain),
  `link.ld`, `startup.c`, `model_f32.c`, `model_q15.c` (Q12 despite its name), `gen.py` (float32 parameters and reference
  outputs from `fastfwd.py`), `qmodel.py` (fixed-point parameters and a bit-exact NumPy emulator of
  `model_q15.c`), `measure.py` (builds every predictor in both forms, checks the outputs, runs the
  simulator, reads the counters, applies Table II of [21]).
- `src/platform_energy.py`, `src/platform_table.py`: verdicts and lifetime ratios on the four processors.
- `src/gap8_lstm_energy.py`: the GAP-8 energy per MAC (imported by `platform_table.py`), its verdicts with
  and without overheads, the per-MAC energies up to which the CNN-LSTM saving stays significant, the
  lowest MACs per cycle at which the taxi saving stays significant, and the 15 fixed-point taxi runs on GAP-8.
- `src/quant_closed_loop.py`: closed-loop test-span suppression of the fixed-point models (emulator)
  against the floating-point models, with the deployed standardisation constants.
- `.gitignore` (excludes `__pycache__/`, `*.pyc` and `run/`).
- `src/quant_energy_check.py`: the paired energy check of the quantisation runs (see Checks).
- `results/m0_measure-O2.json`, `m0_measure-O3.json`, `m0_measure-O3-funroll-loops.json`,
  `results/gap8_lstm_energy.json`, `results/platform_table.json`, `results/qcl_<stream>_<model>.json`
  (six files), `results/quant_energy_check.json`.

Checks
- Every float32 build (alpha = 0.3 for exponential smoothing) reproduces its Python reference (`fastfwd.py` for the two learned models, the
  formulas of `gen.py` for the three cheap predictors) to within 2e-6 on the 16 windows; every fixed-point build
  is bit-exact with `qmodel.py`. Energy per inference varies by less than 0.1% across the seeds and
  streams whose weights were tried.
- Quantisation: in 96 closed-loop runs (CNN-LSTM: intel_time 20 seeds, machine_temp 20, twitter_aapl 20,
  taxi 15, exchange_cpc 1; lightweight: intel_time 20), the fixed-point suppression differs from the
  floating-point value by 0.006 on average and by at most 0.037; the largest int32 accumulator is
  1.66e9 (limit 2.15e9). INT-8, as used by PULP-NN, was not tested. The coverage is uneven: one
  exchange_cpc seed, and the lightweight model on intel_time only.
- Paired energy check (`quant_energy_check.py`): recomputing the energy verdict of each of these runs with
  its fixed-point suppression, on the same seeds, changes no verdict on GAP-8, STM32L4, STM32H7 or the
  Cortex-M0 (with one seed, exchange_cpc is compared by the sign of the net saving). Only these runs are
  checked with quantised forecasts; every other verdict of Table 6 uses floating-point suppression.
- GAP-8: `platform_table.py` and `gap8_lstm_energy.py` give the same verdicts and lifetime ratios; the 15
  fixed-point taxi runs all save on GAP-8 at the four values (smallest saving 4.0 uJ per round at
  0.349 nJ, 2.1 uJ with overheads charged), and are reported separately from the 20 floating-point seeds.
- With 1 nJ per MAC and no overheads, `platform_energy.py` reproduces the verdicts of the paper.
- The counter mapping and the coefficients of Table II of [21] were checked against the published paper,
  and the bit-exactness, energies, verdicts and manuscript numbers were re-derived independently.
- The 0.1% seed and stream variation, the tenfold activation charge and the 10^4-round amortisation were
  checked by editing `measure.py` (weights file), `platform_energy.COST` and `platform_table.H`.
- From a clean unpack on Linux: `build_sim.sh` builds a simulator that reproduces the shipped counters exactly;
  `measure.py` regenerates the three `m0_measure-*.json` byte for byte; `platform_table.py` regenerates
  `platform_table.json`, `gap8_lstm_energy.py` regenerates `gap8_lstm_energy.json` and `quant_energy_check.py`
  regenerates `quant_energy_check.json` byte for byte; `quant_closed_loop.py` regenerates all six `qcl_*.json`
  files byte for byte.

Result (quoted in Section 6.6 and Table 6)
- One CNN-LSTM inference on the Cortex-M0 costs 1.30-2.18 mJ (3.4-5.7 times e_r = 3.8e-4 J; 13-22 nJ
  per MAC), so it cannot pay at the anchor even with complete suppression.
- CNN-LSTM: costs energy significantly on all 11 streams on STM32L4, STM32H7 and the Cortex-M0; on
  GAP-8 saves significantly on taxi at all four values of E_op (significant up to about 0.427 nJ per MAC,
  0.404 with overheads charged) and costs on the other 10.
- Lightweight model: saves on taxi, traffic and exchange_cpc and costs on 8 on GAP-8, STM32L4 and
  STM32H7; on the Cortex-M0 saves on taxi and exchange_cpc in all 30 settings (3 builds x 10
  configurations), on traffic in 10 of 30, and costs on the same 8.

Run order for the added files (inside `run/`)

23. `cp -r ../src/m0 m0 && cd m0 && sh build_sim.sh` (needs network access, git, curl, make, a C compiler and
    libssl-dev; it checks the simulator commit and the toolchain's SHA-256; without network access,
    `cp -r ../src/m0 m0` is still needed for step 26)
24. `OPT="-O2" python3 measure.py`, likewise `OPT="-O3"` and `OPT="-O3 -funroll-loops"` (each in its own
    copy of `m0/`, since the builds share file names) -> `m0_measure-<opt>.json`; copy them to `run/`.
    For example, from `run/`: `mkdir m0_O3 && cp ../src/m0/*.py ../src/m0/*.c ../src/m0/link.ld m0_O3/`; each copy
    must sit next to `m0/` inside `run/`, because `measure.py` reads `../weights/` and `gen.py` and `qmodel.py`
    read `../intel_time.npy` and import `fastfwd.py` (and, in `qmodel.py`, `frozen_benchmark.py`) from `..`. A copy can hold only the
    sources and point to the tools with
    `ARM_GCC_BIN=../m0/xpack-arm-none-eabi-gcc-14.2.1-1.1/bin/ THUMB=../m0/thumb/sim_main THUMBP=../m0/thumbp/sim_main`.
    The README setup copies the shipped `m0_measure-*.json`, `platform_table.json` and `qcl_*.json` into
    `run/`; copy the fresh `m0_measure-*.json` over them before step 25, or `platform_table.py` uses the
    shipped ones.
25. `python3 platform_table.py`                    -> `platform_table.json`
26. `python3 quant_closed_loop.py <stream> <model>` for intel_time cnn_lstm, intel_time lightweight,
    taxi, exchange_cpc, machine_temp and twitter_aapl cnn_lstm -> `qcl_<stream>_<model>.json`
27. `python3 quant_energy_check.py`                -> `quant_energy_check.json` (after steps 25 and 26)
28. `python3 gap8_lstm_energy.py`                  -> `gap8_lstm_energy.json` (after step 26; step 25 imports
    this module for its constants only)

## Part 3: the compute undercount and the untuned baselines quoted in the Introduction and Section 7

The paper says that counting the LSTM input of the CNN-LSTM as one channel rather than the 32 convolution
channels understates its charge by 48%, and that, on taxi at 1 nJ per MAC, this undercount together with an
untuned persistence reference turns a loss into an apparent win, while against the validation-selected
reference the verdict stays a loss at either MAC count. `undercount_check.py` computes these statements from
the saved outputs; nothing is retrained.

What each number is
- The 48% compares the count for that one-channel input, made the same way as the paper's 99,488 MACs
  (12*32*3 + 4*32*(1+32)*12 + 32 = 51,872 MACs), with 99,488: 1 - 51,872/99,488 = 47.9%.
- `src/frozen_benchmark.py` quotes an earlier count, 53,408 MACs, in a module docstring sentence ("raises Einf
  by ~86%") and in a comment ("undercounts by ~46%"). 53,408 is consistent with a bias term also counted as an
  LSTM input (4*32*(1+32+1)*12 + 1,152 + 32); from 51,872 the charge rises by about 92%. Both lines are left
  unchanged, and both undercounts are evaluated. An undercount is a counting error, not another model, so the
  model keeps its 8,481 parameters.
- Setting: taxi, tau = 0.10, anchor e_r = 3.8e-4 J, test span, the 20 floating-point seeds; net saving per
  seed as in `ms_numbers.py`, against untuned persistence (0 MACs) and against the validation-selected
  reference (linear extrapolation, K = 2, as `platform_energy.stream`); Wilcoxon two-sided over the 20 seeds,
  Bonferroni 0.05/22. At 1 nJ per MAC no weight transfer is charged, as in `ms_numbers.py`; at the four GAP-8
  values of Part 2 the weights are charged as there (16-bit, received once, amortised over 1e5 rounds).

Untuned baselines (Section 7): tuning exponential smoothing raised its suppression by up to +0.10 over a
fixed alpha = 0.9 (on rds_cpu_util), and tuning the extrapolation window on taxi raised it from 0.115 (K fixed
at 12) to 0.429. These fixed-parameter values were first produced by an earlier version of
`frozen_benchmark.py` (`results/bench_baselines.json`); `untuned_check.py` recomputes them with the current
evaluator, spans and grids, as `ms_numbers.py` evaluates the cheap arms, on the test span at tau = 0.10.

Added
- `src/undercount_check.py`, `results/undercount_check.json`.
- `src/untuned_check.py`, `results/untuned_check.json`.

Checks
- The result at 1 nJ per MAC, 99,488 MACs, against the validation-selected reference equals `ms_numbers.json`
  (net saving -52.995 uJ per node-round, p = 1.9e-6); at the four GAP-8 values it gives the GAP-8 verdict of
  `gap8_lstm_energy.json` for the CNN-LSTM on taxi (a significant saving).
- From a clean unpack on Linux, `undercount_check.py` regenerates `undercount_check.json` byte for byte.
- `untuned_check.py` asserts that its selected alpha and K and their test suppression equal `ms_numbers.json`
  on every stream; on the ten NAB streams its fixed-parameter values equal those of `bench_baselines.json`
  (taxi K = 12: 0.1146; rds_cpu_util alpha = 0.9: 0.1322; the Intel entry of that file is the withdrawn
  epoch-indexed matrix), and the largest smoothing gain is +0.1008 on rds_cpu_util. From a clean
  unpack on Linux it regenerates `untuned_check.json` byte for byte.

Result (quoted in the Introduction and Section 7)
- At 1 nJ per MAC, against the validation-selected reference, the CNN-LSTM costs energy significantly at
  99,488 MACs (-53.0 uJ per node-round, 0 of 20 seeds saving) and at either undercount (-5.4 uJ at 51,872 and
  -6.9 uJ at 53,408; 1 of 20 seeds saving in each; p = 2.7e-5 and 3.8e-6).
- At 1 nJ per MAC, against untuned persistence, it costs energy significantly at 99,488 MACs (-23.5 uJ) and
  shows a significant but only apparent saving at either undercount (+24.1 and +22.5 uJ; p = 1.9e-6).
- At the GAP-8 values (0.23-0.35 nJ per MAC) the CNN-LSTM saves energy significantly against both references
  at every count, so neither statement holds there; the paper therefore states them at 1 nJ per MAC.

Run order for the added files (inside `run/`)

29. `python3 undercount_check.py`                  -> `undercount_check.json` (imports `platform_energy.py` and
    `gap8_lstm_energy.py`)
30. `python3 untuned_check.py`                     -> `untuned_check.json` (needs the NAB files of the README setup)

## Part 4: figures, and paths of the working directory

No number, data point or result changes. The figures are redrawn for the journal's artwork requirements, and
absolute paths of the machine the runs were made on are replaced.

Changed
- `src/figures.py`, `src/figures_extra.py`, `src/figures_pdf.py` (docstring only in `figures_pdf.py`):
  - Font: Arial throughout, embedded in the PDF files as TrueType; mathematics set in Arial, with variables in
    italics (x, t, f, M, E, e, g, K, alpha, rho, tau) and descriptive subscripts upright (E_sense, E_inf,
    E_op, e_r, rho_ref).
  - One graph per file: the two-panel Figures 2, 3, 6 and 9 are also written one panel per file (suffixes
    `_a` and `_b`), each a copy of the combined figure with the other panel removed. `save_panels` asserts
    that the plotted data, texts, limits, scales and ticks of each panel equal those of the combined figure,
    before the per-panel additions: date tick labels and axis title under Figure 2(a), the span labels over
    Figure 2(b), and the stream names and legend of each Figure 6 panel.
  - Text: no text smaller than 7 pt at the size of the figure, apart from subscripts and superscripts.
  - Colour-independent encoding: circles (lightweight) and triangles (CNN-LSTM) in Figures 5, 6 and 9; line
    styles per stream in Figure 5; the crossovers of Figure 5 are open circles.
  - Figure 8: the cells and the colour bar are drawn as vector shapes (pcolormesh) instead of an embedded
    bitmap; same matrix and orientation.
  - Smaller corrections: Figure 3 ticks at the tested tolerances; Figure 6(b) labelled ticks on both sides of
    zero (symlog axis); points at the edges of Figures 6 and 9 no longer clipped; the panel file of
    Figure 2(a) with its own date axis (in the combined figure the two panels share the axis under (b)).
- `figures/`: every PNG and PDF regenerated with the scripts above, plus the panel files.
- Paths: `RUN_ORDER.md` named the working directory of the runs by its absolute path (line 4); it now says
  "the working directory" (and the dash of its title is a colon). `scripts/bg.sh` and `scripts/bg_intel.sh`
  `cd` to `${WORKDIR:-.}` instead of the absolute path. The five scripts in `audit/` and `audit2/` read and
  write under `WORKDIR` (environment variable, default the current directory, made absolute) instead of the
  absolute path: one added line defines it (it also takes the `import os` of the former `os.chdir` line in
  `audit/recompute.py` and `audit2/cheap.py`) and the path strings use it; no other line changed. The input
  CSV of `audit/stats.py` (`benchmark_run/main_comparison_tau010.csv`, not included) was read from another
  folder of the original machine; the script now looks for it under `WORKDIR`. Nine logs in
  `results/matched_earlier_grids/grid2/` give the source file in their warning lines relative to the
  working directory (`matched/matched_baselines.py`). `SHA256SUMS` and `SHA256SUMS_v1.2` therefore no longer
  match these files; their current hashes are in `SHA256SUMS_v1.4`.

Added
- `src/fig_methodology.py`: Figure 1 (the methodology diagram; a drawing, not generated from data).
- `figures/fig_methodology*` and the panel files `figures/*_a.*`, `figures/*_b.*`.

Checks
- The plotted data are unchanged: running the version 1.3 scripts and these with a recorder that logs every
  array passed to a plotting call gives identical data for all 64 calls of `figures_pdf.py` and all 128 calls
  of `figures_extra.py` (the matrix of Figure 8, drawn with imshow before and with pcolormesh now, included).
  `checkpoint_stats.json` regenerates byte-identical.
- Every PDF in `figures/` embeds only ArialMT, Arial-BoldMT and Arial-ItalicMT and contains no bitmap.
- From a clean unpack on Linux, with matplotlib 3.10.9, Arial version 7.06 and `SOURCE_DATE_EPOCH=1791417600`, steps 31
  to 33 regenerate every PNG and PDF in `figures/` byte for byte.

Run order for the changed and added files (inside `run/`; the setup of the README copies the scripts)

31. `SOURCE_DATE_EPOCH=1791417600 python3 figures_pdf.py`      -> Figures 3, 4, 5, 8 (PNG + PDF, panels of 3)
32. `SOURCE_DATE_EPOCH=1791417600 python3 figures_extra.py`    -> Figures 2, 6, 7, 9 and `checkpoint_stats.json`
33. `SOURCE_DATE_EPOCH=1791417600 python3 fig_methodology.py`  -> Figure 1 (PNG + PDF, panels)
