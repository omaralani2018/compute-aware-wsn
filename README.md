# Energy-Aware Selection of On-Node Predictors for Sustainable Data Reduction in Wireless Sensor Networks

Reproducibility package for the paper by **Omar Bakr Habeeb** and **Mustafa Musa**
(University of Information Technology and Communications, Baghdad).

A predictor is worth running on a sensor node only when the transmission energy its extra
suppression removes exceeds the inference energy it adds:

```
E_inf,cand − E_inf,ref  <  Δρ · e_r                  (the criterion)
e_r*  =  (E_inf,cand − E_inf,ref) / Δρ               (crossover radio cost)
```

Everything here is what produced the reported numbers: the evaluation protocol, the data
preparation, every per-seed output, the saved model weights that exist, and SHA-256 manifests.

This is version 1.5 of the package; see Versions below. `TABLES_AND_FIGURES.md` maps every table, figure
and quoted number of the paper to the command and output file that produce it.

---

## Layout

| Path | Contents |
|---|---|
| `src/` | The protocol and the run scripts (see below) |
| `scripts/` | The background schedulers actually used for the runs (`bg.sh` and `bg_intel.sh` `cd` to the directory the runs were made in, given by `WORKDIR` and by default the current directory; `matched_runall.sh`, `run_lstmdps.sh` and `lstmdps_worker_loop.sh` ran the matched comparison of version 1.2 from the folder holding its scripts) |
| `inputs/` | Input matrices, the ten NAB series, and source hashes |
| `results/` | `learned_*.json`, `tau_*.json` (every seed), `ms_numbers.json`, `routing_*.json`, `checkpoint_stats.json`, `text_checks.json`, `keras_equivalence.json`, `receiver_checks.json`, `epoch_counter.json` and `interpolation_check.json`; from version 1.2, the matched comparison of Table 9: `matched_*.json`, `lstmdps_intel_runs.jsonl`, `matched_summary.json`, `lstmdps_summary.json`, `port_validation.json`, `lstmdps_port_validation.csv` and `matched_earlier_grids/`; from version 1.4, the crossed rule comparison: `rule_crossed_*.json` and `rule_crossed_summary.csv`, and the processor energies of Section 6.5: `m0_measure-*.json`, `gap8_lstm_energy.json`, `platform_table.json`, `qcl_*.json` and `quant_energy_check.json`, and the compute undercount and untuned baselines of the Introduction and Section 7: `undercount_check.json` and `untuned_check.json`; from version 1.5, the robustness analyses of Section 6.9: `sens_*.json`, `intel_variants_coverage.json`, `intel_sensitivity.json`, `v0_check.json`, `intel_mote_blocks.json`, `radio_replay.json` and `effect_intervals.json` |
| `weights/` | Saved model weights, one file per dataset, model and seed |
| `weights_sens/` | Version 1.5. Saved weights of the models retrained on the three Intel preparations of Section 6.9 (one file per preparation, model and seed) |
| `stb/` | Version 1.5. The Grand St Bernard evaluation of Section 6.10: the frozen protocol archive exactly as sent to the co-author (`STB_frozen_2026-10-09.zip`), the files of Amendment 1 (`amendment_1/`), the dated record of both (`FREEZE_RECORD.md`), and the run outputs (`run_outputs/`: per-run results, attempt counts, run log and driver logs, weights, summary) |
| `tm/` | Version 1.5. The training comparison of Section 7.1 (actual against reconstructed history): the frozen protocol archive as sent, the protocol, `TM_FROZEN.sha256`, the code, the run scripts and logs, every per-run result (`tm_runs/`), `tm_summary.json` and a per-stream table with the run record (`TM_RESULTS.md`) |
| `inputs/sensorscope/` | Version 1.5. The SensorScope Grand St Bernard file (`stbernard-meteo.zip`, CC BY 4.0, see Data) |
| `figures/` | Every figure of the paper (Figure 1, the methodology diagram, and the data figures 2–7; also `fig2_cross_dataset` and `fig4_tolerance_sweep`, which earlier drafts used), as PNG (300 dpi) and vector PDF with embedded fonts; the two-panel Figures 1, 2, 3, 5 and 7 also as one file per panel (`_a`, `_b`) |
| `SHA256SUMS` | Hashes of the source files as run |
| `RUN_ORDER.md` | The exact order in which the files were executed |
| `CHANGES_v1.1.md` | What version 1.1 adds, the checks it ran, and the run order of the added files |
| `SHA256SUMS_v1.1` | Hashes of every file added or changed in version 1.1 |
| `MATCHED_BASELINES.md` | Version 1.2: settings, reproduction checks and rerun commands of the matched comparison with three published schemes (Table 9) |
| `CHANGES_v1.2.md` | What version 1.2 adds, the checks it ran, and the run order of the added files |
| `SHA256SUMS_v1.2` | Hashes of every file added or changed in version 1.2 |
| `CHANGES_v1.3.md` | What version 1.3 changes (presentation of Figures 5, 8 and 9) |
| `SHA256SUMS_v1.3` | Hashes of every file changed in version 1.3 |
| `CHANGES_v1.4.md` | What version 1.4 adds (the crossed rule comparison of Section 6.8, the processor energies of Section 6.6, the compute undercount and the untuned baselines of Section 7, and the revised figures), the checks it ran, and the run order of the added files |
| `SHA256SUMS_v1.4` | Hashes of every file added or changed in version 1.4 |
| `CHANGES_v1.5.md` | What version 1.5 adds (the robustness analyses of Section 6.9, the Grand St Bernard evaluation of Section 6.10, the training comparison under `tm/`, the map of tables and figures), the checks it ran, the run order of the added files, and the concordance between the section, table and figure numbers used in earlier files and the current manuscript |
| `SHA256SUMS_v1.5` | Hashes of every file added or changed in version 1.5 |
| `TABLES_AND_FIGURES.md` | Version 1.5. Every table and figure of the paper, and the numbers quoted in Sections 3.2, 6.9 and 6.10, with the command and the output file that produce them |
| `requirements.txt` | Package versions with which the version 1.1 and 1.2 re-runs reproduced the shipped outputs byte for byte on Linux (on Windows, one field of `port_validation.json` differs in its last digits; see `CHANGES_v1.2.md`) |
| `audit/`, `audit2/` | Audit scripts from development, kept for the record. They read and write in the directory the runs were made in, given by `WORKDIR` (by default the current directory); `audit/stats.py` also reads a file that is not included (`benchmark_run/main_comparison_tau010.csv`, kept in another folder when the runs were made and now looked for under `WORKDIR`), so they do not run as shipped |

### `src/`

| File | Role |
|---|---|
| `frozen_benchmark.py` | The protocol: split, tolerance, look-back, epoch cap, checkpoint interval, seeds, window cap, tuning grids, tie-breaks, energy constants, MAC counts |
| `dual_prediction_fix.py` | The closed-loop (shared-state) evaluator and the independent-receiver check |
| `fastfwd.py` | NumPy forward pass, asserted equal to Keras at every checkpoint |
| `prepare_intel.py` | Builds the timestamp-indexed Intel matrix from the raw archive |
| `nab_data.py` | Loads the ten NAB series and their training-span tolerance units |
| `run_learned.py` | Trains and evaluates the learned arms |
| `run_tau.py` | The tolerance sweep |
| `ms_numbers.py` | Computes the numbers reported in the paper into `ms_numbers.json` (the other outputs listed here hold the rest, for example `untuned_check.json` and `undercount_check.json`) |
| `figures.py` | Regenerates Figures 3 and 4 and the two figures earlier drafts used (Figure 3 also as one file per panel) |
| `figures_pdf.py` | Runs `figures.py` and also writes each figure as vector PDF |
| `figures_extra.py` | Regenerates Figures 2, 5, 6 and 7 (PNG and PDF; Figures 2, 5 and 7 also as one file per panel) and the Section 6.8 statistics (`checkpoint_stats.json`) |
| `fig_methodology.py` | Version 1.4. Draws Figure 1, the methodology diagram (PNG and PDF, also one file per panel); not generated from data |
| `routing_baselines.py` | The routing and clustering baselines summarised in Section 3.2 (direct, shortest-path, fuzzy-genetic, LEACH, random multi-hop); the LEACH and direct per-report energies feed Section 5.2 through `text_checks.py` |
| `text_checks.py` | Numbers quoted in the text rather than in a table: the anchor radio cost and its alternatives (Section 5.2), the radio-cost band over which no verdict changes, the open-loop fraction on the Intel test span (Sections 5.3, 6.1), the no-interpolation stress test (needs the raw archive), and, from version 1.5, the energy-budget lifetimes of Section 3.2 |
| `keras_equivalence.py` | Re-checks the NumPy forward pass against Keras on every saved model in `weights/` (needs TensorFlow 2.21) |
| `epoch_counter_check.py` | Why the Intel matrix is indexed by timestamp: backward jumps of the per-mote epoch counter, how far apart readings sharing an epoch number are, and the withdrawn epoch-indexed matrix rebuilt and compared with `inputs/intel_full.npy` (needs the raw archive) |
| `interpolation_check.py` | What the interpolation step of `prepare_intel.py` does to the Intel matrix: cells filled, those from gaps longer than eight bins, and the rows near each split boundary whose interpolated values depend on a reading across it (needs the raw archive) |
| `receiver_checks.py` | Runs the independent-receiver check on the three low-cost arms, with their validation-selected parameters, at every tolerance, on the validation and test spans of all eleven streams |
| `ports_faithful.py` | Version 1.2. Line-by-line Python ports of the MATLAB code released with AM-DR [24]: LMS dual prediction [10] and AM-DR |
| `matched_baselines.py` | Version 1.2. LMS and AM-DR under this protocol, with the released and the single-error transmission rule: selection on validation, one test score, independent receiver |
| `lstmdps_port.py`, `lstmdps_train.py`, `lstmdps_protocol.py` | Version 1.2. LSTM-DPS [25] ported from its released code: transmission loop and NumPy forward pass, training, and the run on the Intel test slice |
| `validate_ports.py` | Version 1.2. Reproduction checks of the three ports against the published figures and the released experiments (needs `raw/data.txt` and, for LSTM-DPS, the authors' repository) |
| `summarise_matched.py` | Version 1.2. The numbers of Table 9 and of the Section 6.7 paragraph |
| `rule_crossed.py` | Version 1.4. Each scheme's two selected parameter sets scored under both transmission rules, with the parameters held fixed (a retrospective diagnostic; Section 6.7) |
| `rule_crossed_check.py` | Version 1.4. Independent scalar re-implementation that re-scores every run of `rule_crossed.py` and asserts identical transmission counts (sums left to right, so it gives the same counts under Python 3.11 and 3.12+) |
| `rule_crossed_summary.py` | Version 1.4. `rule_crossed_summary.csv` and the numbers of the Section 6.7 sentence |
| `m0/` | Version 1.4. The predictors in C (fixed-point and float32) for a Cortex-M0, the build of the simulator of [21] at tag teamplay-D4.5 and of the toolchain (`build_sim.sh`), and `measure.py`, which runs them in the simulator and applies the energy models of [21] |
| `gap8_lstm_energy.py` | Version 1.4. GAP-8 energy per MAC estimated from the cycle counts of a published 16-bit LSTM forecaster [27] (assumed power and frequency; the frequency of the source is ambiguous, see `CHANGES_v1.4.md`), its verdicts with and without activation overheads, the rate margin, and the 15 fixed-point taxi runs on GAP-8 |
| `platform_energy.py`, `platform_table.py` | Version 1.4. Verdicts and lifetime ratios on GAP-8 (energy per MAC from `gap8_lstm_energy.py`), STM32L4 and STM32H7 (energy per MAC from the efficiencies reported in [26]), all estimated from other workloads and not measured for these models, and the simulated Cortex-M0, all with floating-point suppression (Section 6.5; Table 5 shows GAP-8, STM32L4 and the Cortex-M0) |
| `quant_closed_loop.py` | Version 1.4. Closed-loop suppression of the fixed-point models against the floating-point ones |
| `quant_energy_check.py` | Version 1.4. Energy verdicts of the runs of `quant_closed_loop.py` recomputed with their fixed-point suppression, paired with the floating-point ones |
| `untuned_check.py` | Version 1.4. The untuned-baseline numbers quoted in Section 7 (exponential smoothing with alpha fixed at 0.9, linear extrapolation with K fixed at 12), recomputed with the current code |
| `undercount_check.py` | Version 1.4. The compute undercount quoted in the Introduction and Section 7: the 48% (51,872 against 99,488 MACs), the earlier count of 53,408 in a docstring sentence and a comment of `frozen_benchmark.py`, and the taxi verdicts at each count against untuned persistence and the validation-selected reference, at 1 nJ per MAC and at the GAP-8 values |

---

## Reproducing the results

Python 3.11 with NumPy, SciPy, pandas, TensorFlow 2.21 and matplotlib, plus hyperopt and h5py for the
matched comparison of version 1.2. `RUN_ORDER.md` gives the
exact sequence (the steps for the files added in version 1.1 are at the end of `CHANGES_v1.1.md`).

The scripts read their inputs from, and write their outputs to, the directory they are run in. Set up a
run directory first, from the root of this repository:

```bash
mkdir run
cp src/*.py results/*.json inputs/*.npy inputs/*.json inputs/*.csv inputs/mote_locs.txt run/
cp -r weights run/
cd run
python3 -c "import nab_data,os,shutil; [(os.makedirs(os.path.dirname(p),exist_ok=True),shutil.copy('../inputs/nab/'+os.path.basename(p),p)) for p in nab_data.NAB.values()]"
mkdir raw    # put the raw Intel archive here as raw/data.txt (see Data below)
```

The setup copies the shipped results into `run/`, so `run_learned.py` and `run_tau.py` skip every seed that is
already present and change nothing. To retrain, remove `learned_<stream>.json` and `tau_<stream>.json` from
`run/` first, or start from a run directory that holds only the scripts and the inputs. The raw Intel archive
is needed only by `prepare_intel.py`, `text_checks.py --raw`, `epoch_counter_check.py`, `interpolation_check.py`
and `validate_ports.py`.

Then, inside `run/`:

```bash
python3 prepare_intel.py                        # -> intel_time.npy, intel_time_meta.json, intel_time_rows.csv
python3 run_learned.py intel_time lightweight 0 20
python3 run_learned.py intel_time cnn_lstm 0 20
python3 run_tau.py intel_time                   # and likewise for each NAB stream
python3 ms_numbers.py                           # -> ms_numbers.json
export SOURCE_DATE_EPOCH=1791417600             # fixes the creation date stored in the PDF figures
python3 figures_pdf.py                          # Figures 3, 4 and two unused figures (PNG + PDF)
python3 figures_extra.py                        # Figures 2, 5, 6, 7 and checkpoint_stats.json
python3 fig_methodology.py                      # Figure 1 (PNG + PDF)
python3 routing_baselines.py                    # Section 3.2: 30-mote layout, seeds 0-19 -> routing_30motes.json
python3 text_checks.py --raw raw/data.txt       # -> text_checks.json (omit --raw to skip the stress test)
python3 keras_equivalence.py                    # -> keras_equivalence.json
python3 receiver_checks.py                      # -> receiver_checks.json
python3 epoch_counter_check.py                  # -> epoch_counter.json (needs raw/data.txt)
python3 interpolation_check.py                  # -> interpolation_check.json (needs raw/data.txt)
```

Each output can be compared with the copy shipped in `results/`, `inputs/` or `figures/`.

The figure scripts set the font to Arial, embedded in the PDF files as TrueType; Arial is not
redistributed here. Without it, matplotlib uses another font and warns: the plotted data are the same,
but the files are not byte-identical. The shipped figures were written with matplotlib 3.10.9, Arial version 7.06 and
`SOURCE_DATE_EPOCH=1791417600` (which fixes the creation date stored in each PDF); with these, the PNG and
PDF files regenerate byte for byte.

### Figures

| Paper | File | Script |
|---|---|---|
| Figure 1 | `fig_methodology` | `fig_methodology.py` (drawing, not generated from data) |
| Figure 2 | `fig_intel_matrix` | `figures_extra.py` |
| Figure 3 | `fig1_intel_operating_points` | `figures.py` |
| Figure 4 | `fig3_net_saving_vs_radio_cost` | `figures.py` |
| Figure 5 | `fig_per_seed` | `figures_extra.py` |
| Figure 6 | `fig_energy_axis` | `figures_extra.py` |
| Figure 7 | `fig_checkpoints` | `figures_extra.py` |
| (not used) | `fig2_cross_dataset`, `fig4_tolerance_sweep` | `figures.py`; earlier drafts showed them, and their content is in Tables 2 and 4 |

The two-panel figures (1, 2, 3, 5 and 7) are also written one panel per file, with the suffixes `_a` and
`_b` (for example `fig_per_seed_a.pdf`), as the journal asks for one graph per file. For Figures 2, 3, 5 and 7
each panel file is a copy of the combined figure with the other panel removed, and `figures.py` and
`figures_extra.py` assert that its plotted data equal those of the combined figure (before per-panel
additions such as axis labels and legends); `fig_methodology.py`
draws each panel of Figure 1 on its own canvas.

### Routing baselines (Section 3.2)

`routing_baselines.py` is the routing simulator of the original analysis notebook, made standalone.
It is topology-only (every node reports every round, no compute charge) and uses the deployment's
published mote coordinates (`inputs/mote_locs.txt`). On the 30-mote layout of the paper it gives, over
20 seeds, 1,882 rounds for direct, shortest-path and fuzzy-genetic routing, 1,327 ± 1 for LEACH and
831 ± 106 for random multi-hop routing. As a check of the extraction, the withdrawn 33-mote layout of
earlier drafts (`inputs/old33_motes.json`) reproduces their 1,849 rounds for direct and shortest-path
routing: `python3 routing_baselines.py --motes old33_motes.json --seeds 0,1,2,3,4` (inside `run/`). The paper
gives only the direct-transmission lifetime (Section 3.2) and uses the LEACH per-report energy as its anchor
radio cost (Section 5.2); the other routing results are reported here only.

### Matched comparison with published schemes (Table 9)

Version 1.2 runs three published dual-prediction schemes under this protocol: LMS dual prediction
[10] and AM-DR [24], ported from the MATLAB code released with [24], and LSTM-DPS [25], ported from
its released Python code. (Reference numbers are those of the current manuscript; `CHANGES_v1.2.md`, `results/matched_earlier_grids/README.md` and the comments in the version 1.2 source files use the numbering of an earlier draft, in which [13] is [10], [18] is [25] and [25] is [24].) LMS and AM-DR run on all eleven streams with two transmission rules;
LSTM-DPS runs on Intel, five times per mote. `MATCHED_BASELINES.md` gives the sources, the
reproduction of the published figures, the settings, the compute charge and the rerun commands;
the run steps are also at the end of `CHANGES_v1.2.md`. Version 1.4 adds a retrospective crossed
comparison in which each scheme's two selected parameter sets are scored under both rules, so that only
the rule changes, and replaces the per-MAC sensitivity of Section 6.5 by inference energy on four
processors (Section 6.5; three of them in Table 5); the run steps of both are in `CHANGES_v1.4.md`.

### Three routes, and what each one supports

| Route | Coverage |
|---|---|
| **Recompute** every table from the saved per-seed outputs | **All** configurations |
| **Reproduce inference** from saved weights | The seeds present in `weights/`: the CNN-LSTM for all 20 seeds on Intel and on seven NAB streams, 15 on taxi, 1 on exchange_cpc and none on traffic; the lightweight model on Intel only (see the Retrain row). The 17 files `intel_cnn_lstm_s3` to `s19` belong to the withdrawn key `intel` (see Data) and are kept for the record; `keras_equivalence.py` skips them |
| **Retrain** from the input matrices | All configurations. The lightweight model reproduces bit-for-bit; the CNN-LSTM reproduces in distribution only: a re-run of one seed moved test suppression by 0.006 at the same epoch |

Retraining a model is not the same as reproducing inference with its original weights, and the
table above states which route each configuration supports.

### Verification built into the run

Every reported configuration is checked by simulating sender and receiver separately: the sender
emits packets, an independent receiver rebuilds the series from those packets alone, and the check
asserts that the two reconstructions agree to 1e-9 and that the receiver's maximum error never
exceeds the tolerance. It raises rather than returning a flag, and it has passed on every
configuration, including the 880 tolerance-sweep evaluations. For the low-cost arms with their
validation-selected parameters it is re-run at every tolerance by `receiver_checks.py`.

---

## Data

**Intel Berkeley Research Lab.** The raw archive `data.txt` (2,313,682 readings) is not
redistributed here because of its size. It is available from the Intel Lab Data page,
http://db.csail.mit.edu/labdata/labdata.html (reference [22] of the paper); its SHA-256 is in
`inputs/raw_sha256.txt` so a downloaded copy can be checked. `prepare_intel.py` builds a 25,127 × 30 matrix on a 31-second **timestamp**
grid.

Readings are indexed by timestamp, not by the per-mote `epoch` counter. That counter jumps back by
more than a thousand counts during the deployment, mostly at moments shared by many motes, so the same
epoch number recurs days to weeks later and an epoch-indexed pivot places readings taken weeks apart in
the same row (`epoch_counter_check.py`, `results/epoch_counter.json`). `inputs/intel_full.npy` is the withdrawn
epoch-indexed matrix used in earlier drafts; it is kept for the record only and must not be used.

> The dataset key `intel` in `results/` refers to those archived, withdrawn results. The paper's
> Intel results are under the key `intel_time`. Do not re-run under the key `intel`.

**Numenta Anomaly Benchmark.** Ten series, pinned to repository commit `ea702d75`, with file
hashes in `inputs/nab_hashes.txt`. Each is evaluated as one continuous chronological stream; the
tolerance is 0.10 × the standard deviation of that series' training span, so it never sees
validation or test data.

**SensorScope Grand St Bernard** (version 1.5). `inputs/sensorscope/stbernard-meteo.zip`, unmodified, from
G. Barrenetxea, "Sensorscope Data," Zenodo, 2019, doi:10.5281/zenodo.2654726 (reference [31] of the paper),
licensed CC BY 4.0; SHA-256 `2a78ee47f6bc8bab893d1db99ea33aad6b12682a50ee6796379d524464d5f9de`. It was
selected, and the protocol frozen, before any measurement in it was read (`stb/FREEZE_RECORD.md`).

---

## Other files kept for the record

These files are shipped but not used by the steps above:

| File | Contents |
|---|---|
| `results/bench_baselines.json` | Cheap arms of an earlier version of `frozen_benchmark.py`, with alpha fixed at 0.9 and K at 12 (Intel under the withdrawn key `intel_temperature`); `untuned_check.py` recomputes the values the paper quotes |
| `results/bench_intel_clean.json` | An early Intel run on a subset of clean motes (`intel[clean_motes]`), withdrawn |
| `results/equalised_baselines.json`, `results/expsm_alpha_sweep.json` | Earlier tuning comparisons of the low-cost arms (Intel under the withdrawn key `intel`) |
| `results/tau_sweep_results.json` | An earlier summary of the tolerance sweep (Intel under the withdrawn key `intel`); the paper uses `tau_*.json` through `ms_numbers.json` |
| `inputs/intel_A.npy` | A 600-row extract of the withdrawn epoch-indexed Intel matrix |
| `results/bench_intel_time.json` | Cheap arms on `intel_time` from step 2 of `RUN_ORDER.md`; the selection the paper uses is recomputed by `ms_numbers.py` |
| `audit/learned_traffic.orig.json` | A copy of `results/learned_traffic.json`, identical |
| `inputs/nab/NAB_COMMIT.txt` | The pinned commit of the NAB repository |

`RUN_ORDER.md` is the record of the original run (26 September 2026). Some of its paths and files (for example
`intel_raw/raw/data.txt`, `build_ms.py` and `base.docx`) belong to the layout of the machine the runs were made
on; the setup above gives the layout of this package.

## Versions

Version 1.0 (26 September 2026) was the first version; `SHA256SUMS` lists its code as executed.
Version 1.5 adds the analyses of Sections 6.9 and 6.10 and the training comparison under `tm/`; see `CHANGES_v1.5.md`
and `SHA256SUMS_v1.5`. The CHANGES files of versions 1.1 to 1.4, `MATCHED_BASELINES.md`, `RUN_ORDER.md` and the
docstrings of the scripts use the section, table and figure numbers of the drafts they were written for;
`CHANGES_v1.5.md` gives the concordance with the current manuscript.
Versions 1.1 to 1.3 were prepared but not released separately; their changes are part of version 1.4
and are described, with their checks and (where there are any) run steps, in `CHANGES_v1.1.md` to `CHANGES_v1.4.md`, each with its
manifest `SHA256SUMS_v1.1` to `SHA256SUMS_v1.4`.

## Note on the code in this repository

These files are the code as run; the changes made since are listed in this paragraph. Version 1.1 adds files and updates this README; no other existing file is changed (see `CHANGES_v1.1.md`). Version 1.2 does the same, and also adds two lines to `requirements.txt` (see `CHANGES_v1.2.md`). The manuscript generator (`build_ms.py`) is not
included: it reads absolute paths from the machine the runs were performed on and produces the
.docx, which is not part of reproducing the results. `SHA256SUMS` is the manifest of the code as
executed and still lists it. It lists bare file names, so each file is checked from its own folder:
`cd src && sha256sum -c --ignore-missing ../SHA256SUMS` matches seven files and reports `figures.py` and `frozen_benchmark.py` as changed: `figures.py` was revised in version 1.3 (see
`CHANGES_v1.3.md`) and replaced in version 1.4 (see `CHANGES_v1.4.md`), and its current hash is in `SHA256SUMS_v1.4`; version 1.4 shortened one sentence of the module docstring of `frozen_benchmark.py` (no executable line changed), and its current hash is in `SHA256SUMS_v1.4`. The same command in `scripts/` matches
`worker.py` and `tauworker.py` and reports `bg.sh` and `bg_intel.sh` as changed, and `sha256sum -c --ignore-missing SHA256SUMS` at the root reports `RUN_ORDER.md` as changed: version 1.4 replaced the absolute path of the working directory in these three files by a `WORKDIR` setting or a description (no other line changed, apart from the title of `RUN_ORDER.md`, whose dash is now a colon), and their current hashes are in `SHA256SUMS_v1.4`; `build_ms.py` is
the one listed file not included. Where a file appears in more than one manifest, the latest one applies. The files added in version 1.1 are listed with their hashes in `SHA256SUMS_v1.1`, and those added or changed in version 1.2 in `SHA256SUMS_v1.2`. Version 1.3 changes `figures.py`, `figures_extra.py` and three figures (see `CHANGES_v1.3.md` and `SHA256SUMS_v1.3`). Version 1.4 adds files, updates this README, updates the reference numbers in `MATCHED_BASELINES.md`, adds one comment line to `requirements.txt`, corrects two places of `CHANGES_v1.3.md`, shortens one phrase of `CHANGES_v1.2.md` and notes that its reference numbers are those of an earlier draft, corrects the licence note of `LICENSE` (the NAB files are MIT-licensed), shortens one sentence of the module docstring of `frozen_benchmark.py` (no executable line changed), replaces `figures.py`, `figures_extra.py`, `figures_pdf.py` and every file in `figures/` (presentation only; the plotted data are unchanged), and replaces absolute paths of the working directory in `RUN_ORDER.md`, `scripts/bg.sh`, `scripts/bg_intel.sh`, the five scripts in `audit/` and `audit2/` and nine logs in `results/matched_earlier_grids/grid2/` (no computation changed); no other existing file is changed (see `CHANGES_v1.4.md` and `SHA256SUMS_v1.4`). Version 1.5 adds files, updates this README and `LICENSE`, changes the wording of two figure labels in `figures.py` and `fig_methodology.py` (and so ten files in `figures/`), and adds one item to `text_checks.py` (and so one key to `results/text_checks.json`); no other existing file is changed (see `CHANGES_v1.5.md` and `SHA256SUMS_v1.5`).

---

## Citing

```bibtex
@article{habeeb2026energyaware,
  title   = {Energy-Aware Selection of On-Node Predictors for Sustainable
             Data Reduction in Wireless Sensor Networks},
  author  = {Habeeb, Omar Bakr and Musa, Mustafa},
  year    = {2026}
}
```

This version of the package:

```bibtex
@software{habeeb2026computeawarewsn,
  title     = {compute-aware-wsn: reproducibility package},
  author    = {Habeeb, Omar Bakr and Musa, Mustafa},
  version   = {1.5},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23277263},
  url       = {https://github.com/omaralani2018/compute-aware-wsn}
}
```

---

## Licence

Code: MIT (see `LICENSE`). The redistributed datasets remain under their own terms: the Numenta
Anomaly Benchmark files are MIT-licensed by Numenta, Inc. (notice in `inputs/nab/LICENSE_NAB.txt`, from
the pinned commit), the Intel Berkeley data belong to their original publishers, and the SensorScope Grand St Bernard file
(`inputs/sensorscope/`) is licensed CC BY 4.0 by its authors (G. Barrenetxea, Zenodo, doi:10.5281/zenodo.2654726). `src/m0/build_sim.sh` downloads, but the package does not include, the Thumbulator
simulator (MIT) and the xPack Arm GNU toolchain (GPL with the GCC runtime exception; newlib).
