# Tables, figures and quoted numbers: command and output

Numbering is that of the paper this version accompanies (Sections 1 to 7, Tables 1 to 5, Figures 1 to 4) and of its
Supplementary material (Tables S1 to S11, Figures S1 to S3);
`NUMBERING_v1.7.md` maps that of version 1.6. Scripts are in `src/`, except: Grand St Bernard (Section 5.3), in
`stb/STB_frozen_2026-10-09.zip` apart from `effect_intervals.py`, with the corrected `stb_prepare.py` in
`stb/amendment_1/`; the training comparison of Section 6.1 and its supplement, in `tm/`; and those under `caps/`, run from `caps/` after
`caps/setup.sh` as in `caps/README.md`. The others are run inside a run directory set up as in `README.md`. "Intel
preparation steps" and "Grand St Bernard steps" are the run steps in `CHANGES_v1.5.md` (headed there Section 6.9 and
Section 6.10, the numbers of that version). File names such as `table12.json` and `totals_611.json` keep the
numbers of version 1.6. Every output named here is shipped (in `results/`, `stb/run_outputs/`,
`tm/`, `caps/` or `figures/`), so each number can be checked against it without re-running anything; re-running the
command regenerates it.

## Tables

| Table | Command | Output (field) |
|---|---|---|
| 1. Execution cost per forecast and the decisions of CAPS | `caps/avrsim/analyse_avr.py`, `caps/iotlab/analyse_iotlab.py`, `caps/iotlab/fixed/analyse_fixed.py` | `caps/avrsim/avr_summary.json` (`per_forecast`, `streams`), `caps/iotlab/iotlab_<node>_summary.json` and `caps/iotlab/fixed/fixed_m3_21_accuracy_q_summary.json` (also `n_positive`, `n_screened`, `false_exclusions`) |
| 2. CAPS per stream | `caps/screening/screen_bound.py`; `caps/adaptA/oracle_switch.py`, `val_headroom.py`, `gate1_build.py`; `caps/records/records.py` | `caps/records/table12.json`; from `screen_bound.json` (budget, screen), `adaptA/oracle_<stream>_lightweight.json` (`rows[].crit_picks_L`), `val_headroom.json`, `gate1_summary.json` |
| 3. Twelve further datasets | `caps/confA/analyse_conf.py`, `caps/confB/analyse_confB.py`, `caps/confC/analyse_conf.py` | `caps/confA/conf_summary.json` and `caps/confC/conf_summary.json` (`totals`, `ceiling`, `streams.<dataset>`), `caps/confB/confB_summary.json` (`totals`, `ceiling`, `deployments.<deployment>`) |
| 4. What each study measures | none (comparison of the cited papers) | none |
| 5. Published dual-prediction schemes under this protocol | `matched_baselines.py`, `lstmdps_protocol.py`, `summarise_matched.py`; `rule_crossed.py`, `rule_crossed_summary.py` (steps in `CHANGES_v1.2.md`, `CHANGES_v1.4.md`) | `matched_summary.json`, `lstmdps_summary.json`, `rule_crossed_summary.csv` |
| S1. The ten NAB streams | `nab_data.py` (loaded by `ms_numbers.py`) | `ms_numbers.json` (`datasets.<stream>`: `file`, `T`, `unit`) |
| S2. Energy-accuracy operating points (Intel) | `python3 ms_numbers.py` (after `prepare_intel.py`, `run_learned.py`, `run_tau.py`) | `ms_numbers.json` (`datasets.intel_time.by_tau.<tau>`, including `open_loop_frac_train`) |
| S3. Test-span suppression at τ = 0.10 | `run_learned.py`, then `ms_numbers.py` | `learned_<stream>.json`; `ms_numbers.json` (`datasets.<stream>.by_tau."0.1"`, `datasets.<stream>.learned`) |
| S4. The criterion per seed | `ms_numbers.py` | `ms_numbers.json` (`datasets.<stream>.learned.<model>`: difference, p, net saving, crossover and its range) |
| S5. Grand St Bernard | Grand St Bernard steps; `stb_stats.py` | `stb/run_outputs/stb_summary.json` (`primary_obs`); per seed `stb/run_outputs/runs/` |
| S6. Lightweight minus reference at four tolerances | `run_tau.py <stream>`, then `ms_numbers.py` | `tau_<stream>.json`; `ms_numbers.json` (`datasets.<stream>.tau_sweep_lightweight`) |
| S7. Full lifetime ratio on three processors | `ms_numbers.py`; `gap8_lstm_energy.py`; `m0/measure.py`; `platform_table.py` | `ms_numbers.json` (`datasets.<stream>.learned.<model>.L_vs_ref`); `gap8_lstm_energy.json`; `m0_measure-*.json`; `platform_table.json` |
| S8. Intel test span under four preparations | Intel preparation steps; `intel_sens_report.py` | `intel_sensitivity.json` (`preparations`); scored readings from `intel_variants_coverage.json`: `variants.<v>.test.closed_loop_steps` for the three new preparations and `V0.test.cells` for the published matrix; filled share `cells_filled` / `cells` (published matrix: 1 − `cells_observed` / `cells`); per seed `sens_<v>_<model>.json` |
| S9. Training for the protocol | `caps/tolA/analyse_ta.py`, `caps/cltA/analyse_clt.py`; `caps/records/records.py` | `caps/records/table13.json`; `caps/tolA/ta_summary.json`, `caps/cltA/clt_summary.json` (in the latter, `ta_med`, `H2_ta`, `p_h2_ta` and `n_beatR_ta` hold the CLT values; `p_vs_ta` and `d_vs_ta` compare CLT with TA) |
| S10. Four learned models | `caps/midA/analyse_mid.py` | `caps/midA/mid_summary.json` (`verdicts`, `platform_estimate`) |
| S11. Reported data reduction on Intel | values of the cited papers; this work's rows as Tables S2 and S3 | `ms_numbers.json` |

## Figures

| Figure | File in `figures/` | Command |
|---|---|---|
| 1. Methodology | `fig_methodology` (`_a`, `_b`) | `fig_methodology.py` (a drawing, not generated from data) |
| 2. Measured board power of an IoT-LAB M3 node | `caps/iotlab/fig_power_trace` | `caps/iotlab/fig_power_trace.py` (traces `caps/iotlab/data/m3_cons_cut2.txt` and `caps/iotlab/radio/data/radio_m3a_21.oml`) |
| 3. The criterion per seed | `fig_per_seed` (`_a`, `_b`) | `figures_extra.py` |
| 4. Net saving against radio cost | `fig3_net_saving_vs_radio_cost` | `figures_pdf.py` |
| S1. Prepared Intel matrix | `fig_intel_matrix` (`_a`, `_b`) | `figures_extra.py` |
| S2. Persistence against tolerance | `fig1_intel_operating_points` (`_a`, `_b`) | `figures_pdf.py` (runs `figures.py`) |
| S3. Modelled energy | `fig_energy_axis` | `figures_extra.py` |
| Section 5.5, checkpoint selection (not plotted in the paper) | `fig_checkpoints` (`_a`, `_b`) | `figures_extra.py`; the numbers in `checkpoint_stats.json` |

`fig2_cross_dataset` and `fig4_tolerance_sweep` are not in the current manuscript (their content is in
Tables S3 and S6) and are kept for the record.

## Numbers quoted in the text

Numbers are listed by the section that reports them; where the paper gives a summary, the full account is in the
supplement to that section in the Supplementary material.

| Where | Number | Command | Output (field) |
|---|---|---|---|
| Section 3.1 | 1,882 rounds (direct transmission, every node reporting, 0.5 J per node) | `routing_baselines.py` | `routing_30motes.json` (`summary.direct.net_life_mean`) |
| Section 3.1 | about 24,000 rounds with 92.2% suppression at every node | `text_checks.py` | `text_checks.json` (`lifetime_section_3_2`) |
| Section 4.2 | anchor 3.8×10⁻⁴ J (LEACH), about 2.7×10⁻⁴ J (direct), relay values, verdict band | `text_checks.py` | `text_checks.json` (`anchor`, `verdict_band`) |
| Sections 4.3, 5.2 | open-loop fractions | `text_checks.py`, `ms_numbers.py` | `text_checks.json` (`open_loop_test_span`); `ms_numbers.json` |
| Section 5.1 | cycles, energies, gross energies and the streams that stay positive on them, power steps, sample interval, replication within 1% (integer pair in `platform_numbers.json`), integer forecast error, low-cost costs, cycle spread, CAPS per platform, crossover radio cost; receiver check of the integer runs (Section 4.3) | analysis scripts of Table 1; `caps/iotlab/replication.py`; `caps/capsplat/caps_platforms.py`; `caps/records/records.py`; `caps/recv/verify_q.py`, `verify_q_v1.py` | summaries of Table 1; `replication.txt`; `caps_platforms.log`; `caps/records/platform_numbers.json`; `verify_q.log`, `verify_q_v1.log` |
| Section 5.1 | radio cost per report on the nodes' own radios: 2.67×10⁻⁴ J (M3) and 2.66×10⁻⁵ J (nRF52840) for a 4-byte broadcast, 3.25×10⁻⁴ and 4.5×10⁻⁵ J with an acknowledgement, 6.2×10⁻⁴ and 1.08×10⁻⁴ J for 100 bytes; at least 99.7% of frames received; nRF52840 alignment 0.73 against 0.8, its two runs within 1%; at these costs the lightweight model positive on none (M3, software floating point), 2 (M3, integer code: taxi, exchange_cpc) and 1 (nRF52840: taxi) of 11 streams, the CNN-LSTM screened on every stream with no false exclusion, CAPS within 0.6% of its best candidate in hindsight | `caps/iotlab/radio/analyse_radio.py`, `caps/iotlab/radio/radio_decisions.py` | `radio_m3_summary.json`, `radio_nrf_summary.json` (`variants.<v>`: `E_inc_J`, `reports`, `received`; `correlation`), `data/radio_nrf_455113_summary.json`; `radio_decisions.json` (`S_BC/<platform>`: `lightweight_positive_streams`, `screened`, `false_exclusions`, `caps_above_hindsight_pct`) |
| Section 5.1 | end-to-end run: transmit decisions and received readings as emulated (O4, 36 of 36), least costly predictor as predicted (O1, 12 of 12), CAPS choice within 5% of the lowest (O3, 12 of 12), within 15% for 22 of 36 (O2); taxi on the M3 in floating point 217 and 220 µJ per round (reference, lightweight model); M3 11% to 26% above the prediction; nRF52840 low-cost and lightweight within 13%, CNN-LSTM 45% to 48% below; compute-blind selection 12 to 69 times CAPS | `caps/netx/analyse_netx_17a.py`, `analyse_netx_17b.py` (protocol `PREREG17.txt`, Amendments 17a to 17c) | `caps/netx/data/netx_m3f2_summary.json`, `netx_m3i_summary.json`, `netx_nrf4_summary.json` (regenerated in `caps/netx/`; `O1` to `O4`; `cells.<stream>.arms.<arm>`: `E_meas`, `E_pred`; `blind_vs_caps`, measured ratio minus 1); `caps/netx/NETX_RESULTS.txt`; run 455253, descriptive: `caps/netx/data/netx_m3f_DESCRIPTIVE_drift_summary.json` |
| Sections 5.1 to 5.3, Introduction, Discussion | compute-blind selection against CAPS: 10.7% more at 1 nJ per MAC and the anchor (10.4% more than the reference); 34.3%, 10.9%, 5.5% and 3.5% more at 1×10⁻⁴, 3.75×10⁻⁴, 1×10⁻³ and 3×10⁻³ J; 67% more on the nRF52840, 3.7 times (M3, integer code), 16 times (M3, software floating point) and 94 times (ATmega128L) as much; the CNN-LSTM on 58 of 220 seed-stream pairs (all 20 seeds of taxi, 18 of machine_temp); on the twelve further datasets the same energy as CAPS (0.0%), the CNN-LSTM on 7 of 240 pairs, at most 4 of the 20 seeds of any dataset | `caps/blindsel/blind_vs_caps.py`, `blind_conf.py` (protocol `PREREG16.txt`); `caps/blindsel/verify/verify_blind.py`; `caps/blindsel/picks_conf.py` | `blind_vs_caps.json` (`A_1nJ_er<er>`, `B_<platform>`: `tot.blind` / `tot.caps` − 1, `tot.blind` / `tot.ref` − 1, `picks.blind`); `blind_conf.json` (`tot`, `blind_picks`); per stream picks in `verify/verify_out.json` (`median|1nJ|0.00038`: `cnn_picks`); per further dataset `picks_conf.json` (`per_dataset`, `total`, `max_cnn_lstm_seeds_any_dataset`) |
| Introduction, Section 5.1 | one report costs as much as 1,500 to 1,700 MACs (M3, software floating point), 7,700 to 8,300 (M3, integer code) and 3,500 to 4,200 (nRF52840); 380,000 at the anchor and 1 nJ per MAC; the anchor 1.4 and 14 times the report cost | computed: report cost (Section 5.1 row above) divided by the energy per MAC of each learned model (Table 1: energy per forecast over 208 and 99,488 MACs); the anchor divided by the report cost | summaries of Table 1; `radio_m3_summary.json`, `radio_nrf_summary.json` |
| Section 5.1 | integer code with the firmware's single input scale: every CAPS choice unchanged, CAPS total within 0.1% | `caps/records/caps_single_scale.py` | `caps/records/caps_single_scale.json` (`identical_choices`, `caps_vs_ref_pct`) |
| Supplement to Section 5.1 | end-to-end O2 misses within 0.7 points of 15% (15.45% to 15.65%) | none (read from the summaries) | `caps/netx/data/netx_m3f2_summary.json`, `netx_m3i_summary.json` (`cells.<stream>.arms.<arm>`: `E_meas` / `E_pred` − 1) |
| Section 5.2 and Figure S3 note | against the low-cost arm of lowest test-span energy: it differs on five streams, no mean net saving changes sign, the lightweight model on exchange_cpc +7.0 µJ (p = 0.004, not significant) | `caps/records/lowest_energy_arm.py` | `caps/records/lowest_energy_arm.json` (`differs`, `sign_changes`, `streams.exchange_cpc.lightweight`) |
| Section 5.2 (supplement in the Supplementary material) | moving-block bootstrap, computed afterwards: median net saving 27.0, 7.1 and 13.8 µJ per node-round; 95% intervals 16.7 to 38.1 and 14.5 to 39.6 (taxi), −5.8 to 18.3 and −3.6 to 18.9 (traffic), 1.8 to 30.6 and 3.7 to 30.6 (exchange_cpc) with blocks of 48 and 96 rounds | `caps/boot/block_boot.py` | `caps/boot/block_bootstrap.json` (`<stream>`: `point_median_net_uJ`, `B48`, `B96`: `ci95_uJ`, `share_positive`) |
| Section 5.2 | totals: CAPS −0.3%, hindsight −2.0% (−3.0% with switching), always lightweight +11.6%, always CNN-LSTM +57.8%; CNN-LSTM switching at most 0.54% (176 runs with kept weights); sink-assisted switching at 5% on 2 of 11 streams | `caps/records/records.py` | `caps/records/totals_611.json` (`relative`: `caps`, `hind3`, `hind3_switch`, `lw`, `cnn`; `cnn_switching_max_saving`); `caps/adaptA/gate1_summary.json` (`n_pass`). `caps/tolA/verify_ceiling.log` has the totals as first computed, whose hindsight also counts the tolerance-aware model (−3.1% and −3.7%) |
| Section 5.3 | 23 stations, median interval 120 s, period | `stb_prepare.py` | `stb/run_outputs/stb_meta.json` |
| Section 5.3 | 80 runs, receiver checks, two second attempts | `stb_driver.sh` | `stb/run_outputs/runs/*.json`, `*.attempts`, `RUN_LOG.txt` |
| Section 5.3 | 51.9%, predictions, p = 1.9×10⁻⁶, threshold 0.025, per station, per third, filled preparation, GAP-8 values | `stb_stats.py` | `stb/run_outputs/stb_summary.json` |
| Section 5.3 | all 20 seed differences negative in every comparison (sign test) | none (read from the per-seed outputs) | `stb/run_outputs/runs/*.json`; stated in Amendment 1 (`stb/amendment_1/STB_AMENDMENT_1.md`) |
| Section 5.3 | per-station median and interquartile range | `effect_intervals.py` | `effect_intervals.json` (`gsb_per_station`, `gsb_per_third`) |
| Section 5.3 | outcomes C1 to C4, references, screen, headroom flags, CNN-LSTM median net saving of the indoor series; the lightweight model's median net saving per dataset (positive on three NAB streams, at most 18.0 µJ); CAPS at most 1.8% above its best candidate on any dataset | analysis scripts of Table 3 | the summaries of Table 3 (`C1` to `C4`; `streams.<dataset>` or, for SensorScope, `deployments.<deployment>`: `median_net`, `energy.caps` / `energy.hind` − 1) |
| Section 5.3 | the statistic on the thirteen further datasets (the twelve above and Grand St Bernard): Spearman 0.97, mean absolute error 0.062, worst case 0.24; 0.93 and 0.066 over 24; fastness −0.70; −0.31 with the lightweight model's gain | `caps/midA/screen_stat.py` | `screen_stat.json` (`spearman13`, `p_one_sided`, `mae13`, `worst13`, `new`, `lw_delta`); the pooled and fastness values in `screen_stat.log` |
| Section 5.5 | shares of runs selecting the cap or an early checkpoint; validation-test differences | `figures_extra.py` | `checkpoint_stats.json` |
| Section 5.5 | retraining per preparation (seeds 0 to 19, at most 4,000 windows, reselection) | `intel_sens_run.py` | `sens_<v>_<model>.json` (`entries`: `selected_epoch`, `train_windows` per seed); `sens_<v>_cheap.json` (`alpha`, `K`, `selected`) |
| Section 5.5 | initialisation reports under 1% of test readings | `intel_variants.py`, `intel_variant_short100.py` | `intel_variants_coverage.json` (`warmup_cells` / `cells`) |
| Section 5.5 | 0.879 to 0.812; significance in all four preparations; 162,019 common observed readings; ranking of the five arms | `intel_sens_report.py` | `intel_sensitivity.json` (`preparations.<v>`, including `rank_by_suppression`, `rank_by_energy`, `ties_in_suppression`; `common_observed`) |
| Section 5.5 | the published-matrix records reproduce the shipped results | `v0_cells.py` | `v0_check.json` |
| Section 5.5 | replayed net savings (lightweight −11.7 to −25.8 µJ, CNN-LSTM −123.6 to −152.9 µJ per node-round), the same two lightweight seeds saving, effective costs 2.6 to 2.7, 4.3 to 4.6 and 5.9×10⁻⁴ J per extra report (pooled over seeds) | `radio_replay.py` | `radio_replay.json` (`models.<m>.mean`, `seeds_with_positive_net_saving`, `pooled_over_seeds`) |
| Section 5.5 | per-mote differences (mean, range, bootstrap interval) and per third | `intel_mote_blocks.py`, `effect_intervals.py` | `effect_intervals.json` (`intel_per_mote`, `intel_per_third`) |
| Section 5.5 | TA and CLT: streams with a significant increase and with a significant saving; machine_temp p = 0.015 against the Holm threshold of 0.007, 16 of 20 seeds saving; twitter_aapl 0.613 to 0.725 | `caps/tolA/analyse_ta.py`, `caps/cltA/analyse_clt.py` | `ta_summary.json`, `clt_summary.json` (`per_stream.<stream>`: `H1`, `H2_ta`, `p_h2_ta`, medians; `n_h1`, `n_beatR_ta`); seeds in `cltA/clt_machine_temp.json` |
| Section 5.5 | 24-reading window: saves and costs, reference changes; mid-size: counts by rule, medians, saves; the 99,488-MAC CNN-LSTM at 6,440 MACs | `caps/lb24/analyse_lb24.py`, `caps/midA/analyse_mid.py`, `caps/records/records.py` | `lb24_summary.json`; `mid_summary.json` (`counts`, `conclusion`, `stand`); `analyse_mid.log` (medians); `caps/records/cnn_at_6440_macs.json` |
| Section 3.1; supplement to Section 5.4 | 57.8% to 39.5% with sensing at 1×10⁻⁴ J | computed from `caps/records/totals_611.json` (`absolute_J`: reference and CNN-LSTM totals, each plus 11 × (1×10⁻⁴ − 1×10⁻⁷) J) | `totals_611.json` |
| Section 5.5 (full text in the Supplementary material) | open- and closed-loop gap: smaller on 2 of 11, Spearman 0.44 | `caps/shiftA/shift.py` | `shift_summary.json` (`summary.clt`) |
| Supplement to Section 6.1 | training on reconstructed history: inconclusive on all eleven streams, no saving turned into a cost or the reverse | `tm/tm_driver.sh`, `tm/tm_stats.py` (protocol `tm/TM_PROTOCOL.md`) | `tm/tm_summary.json` (`streams.<stream>`: `outcome`, `d_mean`, `d_ci95_bootstrap`, `p_d`, `arms.<arm>.label`); per stream `tm/TM_RESULTS.md`; per run `tm/tm_runs/` |
