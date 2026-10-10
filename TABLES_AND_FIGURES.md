# Tables, figures and quoted numbers: command and output

Numbering is that of the manuscript submitted with this version (Sections 1 to 8, Tables 1 to 11,
Figures 1 to 7). Scripts are in `src/` (those of Section 6.10, apart from `effect_intervals.py`, in `stb/STB_frozen_2026-10-09.zip`, with the corrected
`stb_prepare.py` in `stb/amendment_1/`, and those of Section 7.1 in `tm/`) and are run inside a run directory set up as in `README.md`;
"Section 6.9 steps" and "Section 6.10 steps" are the run steps in `CHANGES_v1.5.md`. Every output named
here is shipped (in `results/`, `stb/run_outputs/`, `tm/` or `figures/`), so each number can be checked
against it without re-running anything; re-running the command regenerates it.

## Tables

| Table | Command | Output (field) |
|---|---|---|
| 1. Energy-accuracy operating points (Intel) | `python3 ms_numbers.py` (after `prepare_intel.py`, `run_learned.py`, `run_tau.py`) | `ms_numbers.json` (`datasets.intel_time.by_tau.<tau>`, including `open_loop_frac_train`) |
| 2. Test-span suppression at τ = 0.10 | `run_learned.py`, then `ms_numbers.py` | `learned_<stream>.json`; `ms_numbers.json` (`datasets.<stream>.by_tau."0.1"`, `datasets.<stream>.learned`) |
| 3. The criterion per seed | `ms_numbers.py` | `ms_numbers.json` (`datasets.<stream>.learned.<model>`: difference, p, net saving, crossover and its range) |
| 4. Lightweight minus reference at four tolerances | `run_tau.py <stream>`, then `ms_numbers.py` | `tau_<stream>.json`; `ms_numbers.json` (`datasets.<stream>.tau_sweep_lightweight`) |
| 5. Full lifetime ratio on three processors | `ms_numbers.py`; `gap8_lstm_energy.py`; `m0/measure.py`; `platform_table.py` | `ms_numbers.json` (`datasets.<stream>.learned.<model>.L_vs_ref`); `gap8_lstm_energy.json`; `m0_measure-*.json`; `platform_table.json` |
| 6. The ten NAB streams | `nab_data.py` (loaded by `ms_numbers.py`) | `ms_numbers.json` (`datasets.<stream>`: `file`, `T`, `unit`) |
| 7. Reported data reduction on Intel | values of the cited papers; this work's rows as Tables 1 and 2 | `ms_numbers.json` |
| 8. What each study measures | none (comparison of the cited papers) | none |
| 9. Published dual-prediction schemes under this protocol | `matched_baselines.py`, `lstmdps_protocol.py`, `summarise_matched.py`; `rule_crossed.py`, `rule_crossed_summary.py` (steps in `CHANGES_v1.2.md`, `CHANGES_v1.4.md`) | `matched_summary.json`, `lstmdps_summary.json`, `rule_crossed_summary.csv` |
| 10. Intel test span under four preparations | Section 6.9 steps; `intel_sens_report.py` | `intel_sensitivity.json` (`preparations`); scored readings from `intel_variants_coverage.json`: `variants.<v>.test.closed_loop_steps` for the three new preparations and `V0.test.cells` for the published matrix; filled share `cells_filled` / `cells` (published matrix: 1 − `cells_observed` / `cells`); per seed `sens_<v>_<model>.json` |
| 11. Grand St Bernard | Section 6.10 steps; `stb_stats.py` | `stb/run_outputs/stb_summary.json` (`primary_obs`); per seed `stb/run_outputs/runs/` |

## Figures

| Figure | File in `figures/` | Command |
|---|---|---|
| 1. Methodology | `fig_methodology` (`_a`, `_b`) | `fig_methodology.py` (a drawing, not generated from data) |
| 2. Prepared Intel matrix | `fig_intel_matrix` (`_a`, `_b`) | `figures_extra.py` |
| 3. Persistence against tolerance | `fig1_intel_operating_points` (`_a`, `_b`) | `figures_pdf.py` (runs `figures.py`) |
| 4. Net saving against radio cost | `fig3_net_saving_vs_radio_cost` | `figures_pdf.py` |
| 5. The criterion per seed | `fig_per_seed` (`_a`, `_b`) | `figures_extra.py` |
| 6. Modelled energy | `fig_energy_axis` | `figures_extra.py` |
| 7. Checkpoint selection | `fig_checkpoints` (`_a`, `_b`) | `figures_extra.py`; the numbers of Section 6.8 in `checkpoint_stats.json` |

`fig2_cross_dataset` and `fig4_tolerance_sweep` are not in the current manuscript (their content is in
Tables 2 and 4) and are kept for the record.

## Numbers quoted in the text

| Where | Number | Command | Output (field) |
|---|---|---|---|
| Section 3.2 | 1,882 rounds (direct transmission, every node reporting, 0.5 J per node) | `routing_baselines.py` | `routing_30motes.json` (`summary.direct.net_life_mean`) |
| Section 3.2 | about 24,000 rounds with 92.2% suppression at every node | `text_checks.py` | `text_checks.json` (`lifetime_section_3_2`) |
| Section 5.2 | anchor 3.8×10⁻⁴ J (LEACH), about 2.7×10⁻⁴ J (direct), relay values, verdict band | `text_checks.py` | `text_checks.json` (`anchor`, `verdict_band`) |
| Sections 5.3, 6.1 | open-loop fractions | `text_checks.py`, `ms_numbers.py` | `text_checks.json` (`open_loop_test_span`); `ms_numbers.json` |
| Section 6.8 | shares of runs selecting the cap or an early checkpoint; validation-test differences | `figures_extra.py` | `checkpoint_stats.json` |
| Section 6.9 | retraining per preparation (seeds 0–19, at most 4,000 windows, reselection) | `intel_sens_run.py` | `sens_<v>_<model>.json` (`entries`: `selected_epoch`, `train_windows` per seed); `sens_<v>_cheap.json` (`alpha`, `K`, `selected`) |
| Section 6.9 | initialisation reports under 1% of test readings | `intel_variants.py`, `intel_variant_short100.py` | `intel_variants_coverage.json` (`warmup_cells` / `cells`) |
| Section 6.9 | 0.879 to 0.812; significance in all four preparations; 162,019 common observed readings; ranking of the five arms | `intel_sens_report.py` | `intel_sensitivity.json` (`preparations.<v>`, including `rank_by_suppression`, `rank_by_energy`, `ties_in_suppression`; `common_observed`) |
| Section 6.9 | the published-matrix records reproduce the shipped results | `v0_cells.py` | `v0_check.json` |
| Section 6.9 | replayed net savings (lightweight −11.7 to −25.8 µJ, CNN-LSTM −123.6 to −152.9 µJ per node-round), the same two lightweight seeds saving, effective costs 2.6–2.7, 4.3–4.6 and 5.9×10⁻⁴ J per extra report (pooled over seeds) | `radio_replay.py` | `radio_replay.json` (`models.<m>.mean`, `seeds_with_positive_net_saving`, `pooled_over_seeds`) |
| Section 6.9 | per-mote differences (mean, range, bootstrap interval) and per third | `intel_mote_blocks.py`, `effect_intervals.py` | `effect_intervals.json` (`intel_per_mote`, `intel_per_third`) |
| Section 6.10 | 23 stations, median interval 120 s, period | `stb_prepare.py` | `stb/run_outputs/stb_meta.json` |
| Section 6.10 | 80 runs, receiver checks, two second attempts | `stb_driver.sh` | `stb/run_outputs/runs/*.json`, `*.attempts`, `RUN_LOG.txt` |
| Section 6.10 | 51.9%, predictions, p = 1.9×10⁻⁶, threshold 0.025, per station, per third, filled preparation, GAP-8 values | `stb_stats.py` | `stb/run_outputs/stb_summary.json` |
| Section 6.10 | all 20 seed differences negative in every comparison (sign test) | none (read from the per-seed outputs) | `stb/run_outputs/runs/*.json`; stated in Amendment 1 (`stb/amendment_1/STB_AMENDMENT_1.md`) |
| Section 6.10 | per-station median and interquartile range | `effect_intervals.py` | `effect_intervals.json` (`gsb_per_station`, `gsb_per_third`) |
| Section 7.1 | training on reconstructed history: inconclusive on all eleven streams, no saving turned into a cost or the reverse | `tm/tm_driver.sh`, `tm/tm_stats.py` (protocol `tm/TM_PROTOCOL.md`) | `tm/tm_summary.json` (`streams.<stream>`: `outcome`, `d_mean`, `d_ci95_bootstrap`, `p_d`, `arms.<arm>.label`); per stream `tm/TM_RESULTS.md`; per run `tm/tm_runs/` |
