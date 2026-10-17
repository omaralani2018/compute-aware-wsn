# Measured Inference Energy and Compute-Aware Predictor Selection for Dual-Prediction Data Reduction in Wireless Sensor Networks

Reproducibility package for the paper by **Omar Bakr Habeeb** and **Mustafa Musa**
(University of Information Technology and Communications, Baghdad), version 1.7.

A predictor is worth running on a sensor node only when the transmission energy its extra suppression removes
exceeds the inference energy it adds:

```
E_inf,cand − E_inf,ref  <  Δρ · e_r                  (the criterion)
e_r*  =  (E_inf,cand − E_inf,ref) / Δρ               (crossover radio cost)
```

The package holds the protocol, the data preparation, every per-seed output, the saved weights and SHA-256
manifests. `TABLES_AND_FIGURES.md` maps every table, figure and quoted number to the command and output that
produce it. `caps/README.md` covers what runs from `caps/`: Sections 4.4 and 5.1 and parts of Sections 5.2, 5.3,
5.5 and 6.1. Numbers are those of the manuscript this version accompanies; `NUMBERING_v1.7.md` maps those of version
1.6.

## Layout

| Path | Contents |
|---|---|
| `src/` | protocol and run scripts (below) |
| `scripts/` | schedulers used for the runs (`WORKDIR` sets the run directory) |
| `inputs/` | input matrices, the ten NAB series and their hashes; `inputs/sensorscope/`, the Grand St Bernard file |
| `results/` | per-seed outputs (`learned_*.json`, `tau_*.json`) and the outputs of every analysis script |
| `weights/`, `weights_sens/` | saved weights, one file per dataset, model and seed (`weights_sens/`: the Intel preparations of Section 5.5) |
| `stb/` | Grand St Bernard evaluation (Section 5.3): frozen protocol archive, Amendment 1, `FREEZE_RECORD.md`, outputs |
| `tm/` | training comparison of Section 6.1 and its supplement: protocol, code, per-run results, `TM_RESULTS.md` |
| `caps/` | CAPS, the processor and radio measurements, the twelve further datasets, the training, look-back and model-size analyses and the open- and closed-loop comparison (Sections 4.4, 5.1 to 5.3, 5.5 and 6.1) |
| `caps/blindsel/`, `caps/netx/`, `caps/boot/` | added in 1.7: compute-blind selection against CAPS (Sections 4.5 and 5.1 to 5.3); dual prediction run end to end on the nodes (Sections 4.4 and 5.1); moving-block bootstrap over test rounds (Section 5.2, computed afterwards) |
| `figures/` | Figures 1, 3 and 4 and Figures S1 to S3 as PNG (300 dpi) and PDF, two-panel figures also per panel (`_a`, `_b`), the checkpoint figure summarised in Section 5.5 and two figures of earlier drafts; Figure 2 is in `caps/iotlab/` |
| `RUN_ORDER.md` | order of the original run (26 September 2026) |
| `MATCHED_BASELINES.md` | matched comparison of Table 5: sources, reproduction checks, settings, rerun commands (older numbering) |
| `CHANGES_v1.1.md` to `CHANGES_v1.7.md` | what each version added or changed, its checks and run steps; `CHANGES_v1.5.md` maps the numbers of version 1.4 files to those of version 1.5 |
| `NUMBERING_v1.7.md` | section, table, figure and reference numbers of version 1.6 docs and of this version |
| `SHA256SUMS*` | manifests (see Versions and manifests) |
| `requirements.txt` | package versions of the re-runs |
| `audit/`, `audit2/` | development audit scripts, kept for the record; they need `WORKDIR` and a file not included |

### `src/`

| File | Role |
|---|---|
| `frozen_benchmark.py` | the protocol: split, tolerance, look-back, epochs, checkpoints, seeds, grids, tie-breaks, energy constants, MAC counts |
| `dual_prediction_fix.py` | closed-loop evaluator and independent-receiver check |
| `fastfwd.py` | NumPy forward pass, asserted equal to Keras at every checkpoint |
| `prepare_intel.py`, `nab_data.py` | Intel matrix from the raw archive; the ten NAB series and their tolerance units |
| `run_learned.py`, `run_tau.py` | learned arms; tolerance sweep |
| `ms_numbers.py` | the paper's numbers into `ms_numbers.json` |
| `figures.py`, `figures_pdf.py`, `figures_extra.py`, `fig_methodology.py` | the figures (see Figures) and `checkpoint_stats.json` |
| `routing_baselines.py` | routing and clustering baselines of Section 3.1 |
| `text_checks.py` | numbers quoted in the text: anchor radio cost and its band, open-loop fraction, stress test, lifetimes of Section 3.1 |
| `keras_equivalence.py`, `receiver_checks.py` | NumPy against Keras on every saved model; receiver check of the low-cost arms at every tolerance |
| `epoch_counter_check.py`, `interpolation_check.py` | why Intel is indexed by timestamp; what interpolation fills (need the raw archive) |
| `ports_faithful.py`, `matched_baselines.py`, `lstmdps_*.py`, `validate_ports.py`, `summarise_matched.py` | the matched comparison of Table 5: LMS [15] and AM-DR [41] ported from the code of [41], LSTM-DPS [42] from its code, their reproduction checks and summary |
| `rule_crossed*.py` | crossed comparison of the two transmission rules (Section 5.6) |
| `m0/` | predictors in C for a Cortex-M0 and the simulator of [34] (`build_sim.sh` needs about 1.3 GB) |
| `gap8_lstm_energy.py`, `platform_energy.py`, `platform_table.py` | processor energies of Section 5.4 (GAP-8 from [47], STM32L4 and STM32H7 from [46], Cortex-M0 simulated) |
| `quant_closed_loop.py`, `quant_energy_check.py` | fixed-point suppression and its energy verdicts |
| `untuned_check.py`, `undercount_check.py` | untuned baselines and the 48% compute undercount (Introduction, Section 6) |
| `intel_variants.py`, `intel_variant_short100.py`, `segeval.py`, `test_segeval.py`, `intel_sens_run.py`, `v0_cells.py`, `intel_sens_report.py`, `intel_mote_blocks.py`, `radio_replay.py`, `effect_intervals.py` | analyses of Section 5.5 and intervals of Section 5.3 (run order in `CHANGES_v1.5.md`) |

## Reproducing the results

Python 3.11 with NumPy, SciPy, pandas, TensorFlow 2.21 and matplotlib (plus hyperopt and h5py for Table 5).
`caps/` has its own setup and steps (`caps/README.md`). From the root of this package:

```bash
mkdir run
cp src/*.py results/*.json results/*.jsonl inputs/*.npy inputs/*.json inputs/*.csv inputs/mote_locs.txt run/
cp -r weights src/m0 run/
cd run
python3 -c "import nab_data,os,shutil; [(os.makedirs(os.path.dirname(p),exist_ok=True),shutil.copy('../inputs/nab/'+os.path.basename(p),p)) for p in nab_data.NAB.values()]"
mkdir raw    # put the raw Intel archive here as raw/data.txt (see Data below)
```
The shipped results are copied into `run/`, so the training scripts skip every seed already present; to retrain,
remove `learned_<stream>.json` and `tau_<stream>.json` first. The raw Intel archive (`raw/data.txt`) is needed only by
`prepare_intel.py`, `text_checks.py --raw`, `epoch_counter_check.py`, `interpolation_check.py`, `validate_ports.py`,
`v0_cells.py` and the Section 5.5 scripts that rebuild the Intel preparations. `stb_stats.py` (in `stb/STB_frozen_2026-10-09.zip`), `intel_sens_report.py` and
the per-reading analyses of Sections 5.3 and 5.5 (Tables S5 and S8) also need the per-reading records of version 1.5
(`cells/`, see Versions and manifests) or the raw file through `v0_cells.py`. Inside `run/`:

```bash
python3 prepare_intel.py                        # -> intel_time.npy, intel_time_meta.json, intel_time_rows.csv
python3 run_learned.py intel_time lightweight 0 20
python3 run_learned.py intel_time cnn_lstm 0 20
python3 run_tau.py intel_time                   # and likewise for each NAB stream
python3 ms_numbers.py                           # -> ms_numbers.json
export SOURCE_DATE_EPOCH=1791417600             # fixes the creation date stored in the PDF figures
python3 figures_pdf.py                          # Figures S2, 4 and two unused figures (PNG + PDF)
python3 figures_extra.py                        # Figures S1, 3, S3, the checkpoint figure and checkpoint_stats.json
python3 fig_methodology.py                      # Figure 1 (PNG + PDF)
python3 routing_baselines.py                    # Section 3.1: 30-mote layout, seeds 0 to 19 -> routing_30motes.json
python3 text_checks.py --raw raw/data.txt       # -> text_checks.json (without --raw: no stress test, and that entry is dropped)
python3 keras_equivalence.py                    # -> keras_equivalence.json
python3 receiver_checks.py                      # -> receiver_checks.json
python3 epoch_counter_check.py                  # -> epoch_counter.json (needs raw/data.txt)
python3 interpolation_check.py                  # -> interpolation_check.json (needs raw/data.txt)
```
Each output can be compared with the shipped copy. The figures regenerate byte for byte with matplotlib 3.10.9,
Arial 7.06 (not redistributed) and the `SOURCE_DATE_EPOCH` above.

### Figures

| Paper | File | Script |
|---|---|---|
| Figure 1 | `fig_methodology` | `fig_methodology.py` (drawing, not generated from data) |
| Figure S1 | `fig_intel_matrix` | `figures_extra.py` |
| Figure 2 | `caps/iotlab/fig_power_trace` | `caps/iotlab/fig_power_trace.py` (see `caps/README.md`) |
| Figure S2 | `fig1_intel_operating_points` | `figures_pdf.py` (runs `figures.py`) |
| Figure 3 | `fig_per_seed` | `figures_extra.py` |
| Figure S3 | `fig_energy_axis` | `figures_extra.py` |
| Figure 4 | `fig3_net_saving_vs_radio_cost` | `figures_pdf.py` (runs `figures.py`) |
| (Section 5.5, not plotted in the paper) | `fig_checkpoints` | `figures_extra.py` |
| (not used) | `fig2_cross_dataset`, `fig4_tolerance_sweep` | `figures.py`; earlier drafts showed them, and their content is in Tables S3 and S6 |

The routing simulator (`routing_baselines.py`) gives, over 20 seeds on the 30-mote layout, 1,882 rounds for direct,
shortest-path and fuzzy-genetic routing, 1,327 ± 1 for LEACH and 831 ± 106 for random multi-hop; the paper uses the
direct lifetime and the LEACH per-report energy. The matched comparison is documented in `MATCHED_BASELINES.md`,
`CHANGES_v1.2.md` and `CHANGES_v1.4.md`.

### What each route reproduces

| Route | Coverage |
|---|---|
| Recompute every table from the per-seed outputs | all configurations |
| Inference from saved weights | the CNN-LSTM for all 20 seeds on Intel and seven NAB streams, 15 on taxi, 1 on exchange_cpc, none on traffic; the lightweight model on Intel and, in `caps/adaptA/weights`, the ten NAB streams (retrained with the frozen protocol). The files `intel_cnn_lstm_s3` to `s19` belong to the withdrawn key `intel` |
| Retrain from the input matrices | all configurations; the lightweight model bit for bit, the CNN-LSTM in distribution (one seed moved by 0.006) |

Every reported configuration passes the independent-receiver check: the receiver rebuilds the series from the packets
alone, agrees with the sender to 1e-9 and never exceeds the tolerance. The check runs its own sender and receiver with
the same predictor, so it tests the closed-loop reconstruction; the reported suppression rates are reproduced from
the saved weights (row above).

## Data

- **Intel Berkeley Research Lab** [37]: `data.txt` (2,313,682 readings) is not redistributed; get it from
  http://db.csail.mit.edu/labdata/labdata.html (copy used: GitHub mirror `linsea423/Intel_Lab_Data`) and check it
  against `inputs/raw_sha256.txt`. `prepare_intel.py` builds a 25,127 × 30 matrix on a 31-s timestamp grid; the
  per-mote epoch counter jumps back and is not used. Results under the key `intel` (and `inputs/intel_full.npy`)
  are the withdrawn epoch-indexed version; the paper uses `intel_time`.
- **Numenta Anomaly Benchmark**: ten series at commit `ea702d75` (hashes in `inputs/nab_hashes.txt`); the tolerance
  is 0.10 × the standard deviation of the training span.
- **SensorScope Grand St Bernard** [43]: `inputs/sensorscope/stbernard-meteo.zip`, unmodified (G. Barrenetxea, Zenodo,
  doi:10.5281/zenodo.2654726, CC BY 4.0), SHA-256 `2a78ee47f6bc8bab893d1db99ea33aad6b12682a50ee6796379d524464d5f9de`;
  the protocol was frozen before any measurement in it was read (`stb/FREEZE_RECORD.md`).
- **The twelve further datasets of Section 5.3**: in `caps/confA/data/` and `caps/confB/data/`; the indoor data are not redistributed
  (see `caps/README.md`).

### Files kept for the record

| File | Contents |
|---|---|
| `results/bench_baselines.json` | Low-cost arms of an earlier version of `frozen_benchmark.py`, with alpha fixed at 0.9 and K at 12 (Intel under the withdrawn key `intel_temperature`); `untuned_check.py` recomputes the values the paper quotes |
| `results/bench_intel_clean.json` | An early Intel run on a subset of clean motes (`intel[clean_motes]`), withdrawn |
| `results/equalised_baselines.json`, `results/expsm_alpha_sweep.json` | Earlier tuning comparisons of the low-cost arms (Intel under the withdrawn key `intel`) |
| `results/tau_sweep_results.json` | An earlier summary of the tolerance sweep (Intel under the withdrawn key `intel`); the paper uses `tau_*.json` through `ms_numbers.json` |
| `inputs/intel_A.npy` | A 600-row extract of the withdrawn epoch-indexed Intel matrix |
| `results/bench_intel_time.json` | Low-cost arms on `intel_time` from step 2 of `RUN_ORDER.md`; the selection the paper uses is recomputed by `ms_numbers.py` |
| `audit/learned_traffic.orig.json` | A copy of `results/learned_traffic.json`, identical |
| `inputs/nab/NAB_COMMIT.txt` | The pinned commit of the NAB repository |

`RUN_ORDER.md` uses the layout of the machine the runs were made on; the setup above gives the layout here.

## Versions and manifests

Version 1.0 (26 September 2026) was the first; 1.1 to 1.3 were folded into 1.4; 1.5 added the Intel preparations
(Section 5.5), Grand St Bernard (Section 5.3) and `tm/`; 1.6 added `caps/`; 1.7 adds `caps/blindsel/`,
`caps/netx/` and `caps/boot/`. The per-reading records of the Intel preparations and Grand St Bernard
(`compute-aware-wsn-v1.5-cells.zip`) are in the version 1.5 record, doi:10.5281/zenodo.23277263.

- `SHA256SUMS`: the code as run in version 1.0 (bare names; check from each folder). `build_ms.py`, the manuscript
  generator, is listed but not included.
- `SHA256SUMS_v1.x`: the files added or changed in that version. Where a file is in several manifests, the latest
  applies, so older manifests report later-changed files as changed; each CHANGES file says which.
- `SHA256SUMS_all`: every file except itself and `SHA256SUMS_v1.7`, plus the version 1.5 cells archive; check it, and
  `SHA256SUMS_v1.5`, with `sha256sum -c --ignore-missing`.

Files of earlier versions, including docstrings, keep the section and reference numbers of their drafts:

| Work | This version | Version 1.6 docs | `MATCHED_BASELINES.md`, `CHANGES_v1.4.md`, `CHANGES_v1.5.md` and this README before version 1.6 | `CHANGES_v1.2.md`, `results/matched_earlier_grids/README.md` and comments of the version 1.2 sources |
|---|---|---|---|---|
| Santini and Römer (LMS dual prediction) | [15] | [18] | [10] | [13] |
| Fathy et al. (AM-DR) | [41] | [34] | [24] | [25] |
| Fathalla et al. (LSTM-DPS) | [42] | [35] | [25] | [18] |
| Nikov et al. (Cortex-M0 simulator) | [34] | [29] | [21] | |
| Garofalo et al. (PULP-NN) | [46] | [36] | [26] | |
| Cappellone et al. (GAP-8 LSTM forecaster) | [47] | [37] | [27] | |
| Bodik et al. (Intel Lab Data) | [37] | [32] | [22] | |
| Morales et al. (deep learning in a dual prediction scheme) | [49] | [40] | [30] | |
| Barrenetxea (Sensorscope Data) | [43] | [41] | [31] | |
| Habeeb and Musa (this package) | [50] | [45] | [32] | |

`NUMBERING_v1.7.md` maps the section, table and figure numbers of version 1.6 docs to this version.

## Citing

```bibtex
@article{habeeb2026computeaware,
  title   = {Measured Inference Energy and Compute-Aware Predictor Selection for
             Dual-Prediction Data Reduction in Wireless Sensor Networks},
  author  = {Habeeb, Omar Bakr and Musa, Mustafa},
  year    = {2026}
}
```

This version of the package:

```bibtex
@software{habeeb2026computeawarewsn,
  title     = {compute-aware-wsn: reproducibility package},
  author    = {Habeeb, Omar Bakr and Musa, Mustafa},
  version   = {1.7},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23440217},
  url       = {https://github.com/omaralani2018/compute-aware-wsn}
}
```

---

## Licence

Code and documentation: MIT. Outputs (results, figures, weights, run records): CC BY 4.0 (see `LICENSE`).
Third-party data keep their terms: NAB files MIT (Numenta, Inc., `inputs/nab/LICENSE_NAB.txt`); Intel data with
acknowledgement, as the Intel Lab Data page states; SensorScope files CC BY 4.0 (G. Barrenetxea). The flash images in
`caps/iotlab/firmware/`, `caps/iotlab/fixed/firmware/`, `caps/iotlab/radio/firmware/` and `caps/netx/firmware/`
contain RIOT (LGPL-2.1), Arm CMSIS (Apache-2.0) and newlib, with the application source in `caps/iotlab/` and
`caps/netx/`; `caps/avrsim/fw/fw.elf` contains
avr-libc (BSD) and a section of simavr (GPL-3.0). `src/m0/build_sim.sh` downloads, but the package does not include,
the Thumbulator simulator (MIT) and the xPack Arm GNU toolchain. See `LICENSE`,
`caps/iotlab/firmware/THIRD_PARTY_NOTICES.txt` and `caps/netx/firmware/THIRD_PARTY_NOTICES.txt`.
