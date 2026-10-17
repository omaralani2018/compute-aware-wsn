# caps/: CAPS, the confirmatory evaluations and the processor and radio measurements (version 1.7)

What produced Sections 4.4 and 5.1 (processor, radio and end-to-end measurements), CAPS and its totals in Section 5.2,
the twelve further datasets and the test of the screening statistic in Section 5.3, the training, look-back and
model-size analyses of Section 5.5, the comparison with compute-blind selection (Sections 4.5 and 5.1 to 5.3) and the
comparison of open- and closed-loop suppression in the supplement to Section 6.1. Folder names are those used in the runs, because the
scripts find each other by them. Section, table and figure numbers are those of the manuscript this version
accompanies; `../NUMBERING_v1.7.md` maps those of version 1.6.

## Terms

| Term | Meaning |
|---|---|
| CAPS | Algorithm 1: Screen (exclude before training a candidate whose extra inference energy exceeds the largest radio saving it could achieve), Select (highest net saving on validation, or the reference), Stay-or-Switch (could run-time switching pay; the adaptivity-headroom test of `adaptA/PREREG4.txt`) |
| reference, low-cost arms | persistence, exponential smoothing, linear extrapolation; the reference has the highest validation suppression |
| lightweight model, CNN-LSTM | the learned predictors of 208 and 99,488 multiply-accumulate operations (MACs) |
| headroom | saving of hindsight block switching (blocks of 96 readings) over the better static choice |
| Gate 1, Gate 2 | the two steps of the sink-assisted switching study (`adaptA/PREREG.txt`, `PREREG2.txt`) |
| TA, CLT | tolerance-aware and closed-loop tolerance-aware training (Section 5.5) |
| compute-blind selection | per seed, the highest validation suppression among the reference and the two learned models, with no screen (Section 4.5; `blindsel/PREREG16.txt`) |
| O1 to O4 | the outcomes of the end-to-end run: ranking, accuracy within 15%, CAPS on the node, loop (`netx/PREREG17.txt`) |
| fastness | Eq. 7 of the paper |

## Folders

| Folder | Paper | Contents |
|---|---|---|
| `screening/` | 5.2, Table 2 | screening budget per stream and whether an excluded candidate would have saved |
| `adaptA/` | 5.2, Table 2; sink-assisted switching | block switching, validation headroom, retrained NAB lightweight models, Gate 1, protocols `PREREG.txt`, `PREREG2.txt`, `PREREG4.txt` |
| `tolA/` | 5.5, Table S9 (TA); 5.2 totals | TA training and analysis, `PREREG3.txt`, weights; `verify_ceiling.py` and `caps_vs_fixed.py` (totals) |
| `cltA/` | 5.5, Table S9 (CLT) | CLT training and analysis, `PREREG5.txt`, weights |
| `shiftA/` | 6.1 | open- and closed-loop gap, `PREREG8.txt` |
| `confA/` | 5.3, Table 3 (six NAB streams) | data, training, analysis, `PREREG6.txt`, weights |
| `confB/` | 5.3, Table 3 (four SensorScope deployments) | data, evaluator copies in `code/`, analysis, `PREREG7.txt`, `AMENDMENT_7a.txt`, per-deployment runs |
| `confC/` | 5.3, Table 3 (two indoor datasets) | training, analysis, `PREREG10.txt`, data hashes (data not redistributed) |
| `avrsim/` | 4.4, 5.1, Table 1 (ATmega128L) | firmware, simulator output, analysis, `PREREG11.txt` |
| `iotlab/` | 4.4, 5.1, Table 1 (IoT-LAB M3 and nRF52840, floating point); Figure 2 | RIOT application `app/`, flash images `firmware/`, host check, data of four nodes, full consumption records `records/`, analysis, `PREREG12.txt`; `fig_power_trace.py` (Figure 2) |
| `iotlab/fixed/` | 4.4, 5.1, Table 1 (M3, integer code) | emulators, application `app_q/`, flash image, data, analysis, `PREREG13.txt` |
| `iotlab/radio/` | 4.4, 5.1 (radio cost per report) | RIOT application (`main.c`, `Makefile`), flash images `firmware/` (`firmware/v1/`: first version), analysis, decisions, `PREREG15.txt`, `AMENDMENT_15a.txt`, first-version files `v1/`, data and logs `data/` |
| `netx/` | 4.4, 5.1 (end-to-end run) | generators, RIOT applications `app/` (sender and sink) and `app_rx2/` (sink v2), flash images `firmware/`, host check `check/`, analyses, data `data/`, `PREREG17.txt`, `AMENDMENT_17a.txt` to `AMENDMENT_17c.txt`, `RUNS_LOG.txt`, `NETX_RESULTS.txt` |
| `capsplat/` | 5.1 | CAPS per platform |
| `blindsel/` | 4.5, 5.1 to 5.3 | compute-blind selection against CAPS, `PREREG16.txt`; independent recomputation `verify/`; picks per dataset `picks_conf.py` (Records); first draft of the protocol `superseded/` |
| `recv/` | 4.3 | receiver check of the integer-code runs |
| `lb24/` | 5.5 (24-reading window) | runs, analysis, `PREREG9.txt` |
| `midA/` | 5.5, Table S10; 5.3 (statistic) | mid-size models, statistic on further datasets, `PREREG14.txt`; `verdict_rule_check.py` (Deviations, item 14) |
| `records/` | 5.1, 5.2, 5.5 | numbers printed or copied by hand, written to files (Records) |
| `boot/` | 5.2 (computed afterwards) | moving-block bootstrap over the test rounds of taxi, traffic and exchange_cpc: `block_boot.py`, `block_bootstrap.json` |

## Prerequisites and setup

Python 3.11 with the versions of `requirements.txt` at the root (checked with Python 3.11.15, NumPy 2.4.4, SciPy
1.17.1, pandas 3.0.2, matplotlib 3.10.9, TensorFlow 2.21.0, Keras 3.15.1). For the firmware (Ubuntu 24.04):
`gcc-avr` 7.3.0, `avr-libc` 2.0.0, `simavr` and `libsimavr-dev` 1.6, `gcc-arm-none-eabi` 13.2.1,
`libnewlib-arm-none-eabi` 4.4.0, `qemu-user` 8.2.2 (for `netx/check/`), `gcc`, `make`, `git`, and RIOT (see Firmware). Only `confC/analyse_conf.py` and
`midA/screen_stat.py` need the indoor data (see Data).

From this folder, `bash setup.sh` (about 25 s) builds `stage17/run` and `stb_exec`, prepares the SensorScope
deployments (checked against `confB/SEGMENTS.sha256`) and makes the links the scripts expect.

## Running

Each line runs from `caps/`; every analysis rewrites its summary with an identical file (keep the order: the first
nRF52840 radio line writes `radio_nrf_summary.json` with run 455113's values and the next restores it). Run in a
copy of the package to keep the shipped files apart. The block takes about 10 minutes on one core. Training is not needed to check any
number.

```bash
(cd screening && python3 -I screen_bound.py ../stage17/run/ms_numbers.json)    # screen_bound.json
(cd adaptA && python3 oracle_switch.py taxi lightweight)    # oracle_taxi_lightweight.json; any stream, lightweight or cnn_lstm
(cd adaptA && python3 gate1_build.py)                        # gate1_summary.json
(cd adaptA && python3 val_headroom.py)                       # val_headroom.json
(cd adaptA && python3 verify_oracle.py)                      # about 5 s; prints FAILURES: 0
(cd tolA && python3 -I analyse_ta.py)                        # ta_summary.json
(cd tolA && python3 recheck.py)                              # prints mismatches 0
(cd cltA && python3 -I analyse_clt.py)                       # clt_summary.json (see the note below)
(cd cltA && python3 recheck_clt.py)                          # prints mismatches 0
(cd shiftA && python3 shift.py)                              # shift_summary.json, about 20 s
(cd confA && python3 -I analyse_conf.py)                     # conf_summary.json
(cd confB && python3 -I analyse_confB.py)                    # confB_summary.json, about 3 min
(cd confC && python3 -I analyse_conf.py)                     # conf_summary.json (needs the indoor data)
(cd lb24 && python3 analyse_lb24.py)                         # lb24_summary.json
(cd midA && python3 analyse_mid.py)                          # mid_summary.json
(cd midA && python3 screen_stat.py)                          # screen_stat.json (needs the indoor data)
(cd avrsim && python3 analyse_avr.py)                        # avr_summary.json
(cd iotlab && python3 analyse_iotlab.py data/m3_serial.txt data/m3_cons_cut2.txt m3)      # iotlab_m3_summary.json
(cd iotlab && python3 analyse_iotlab.py data/m3b_serial.txt data/m3b_cons_cut.txt m3b)
(cd iotlab && python3 analyse_iotlab.py data/nrf_serial.txt data/nrf_cons_cut.txt nrf)
(cd iotlab && python3 analyse_iotlab.py data/nrfb_serial.txt data/nrfb_cons_cut.txt nrfb)
(cd iotlab && python3 replication.py > replication.txt)
(cd iotlab/fixed && python3 accuracy_q.py)                   # accuracy_q.json, about 1 min
(cd iotlab/fixed && python3 analyse_fixed.py data/fix_serial.txt m3-21 data/fix_m3_21.txt m3_21_accuracy_q accuracy_q.json)
(cd iotlab/fixed && python3 analyse_fixed.py data/fix_serial.txt m3-22 data/fix_m3_22.txt m3_22_accuracy_q accuracy_q.json)
(cd iotlab/fixed && python3 analyse_fixed.py data/fix_serial.txt m3-21 data/fix_m3_21.txt m3_21_accuracy_q_v1_fixed_si20 accuracy_q_v1_fixed_si20.json)
(cd iotlab/fixed && python3 analyse_fixed.py data/fix_serial.txt m3-22 data/fix_m3_22.txt m3_22_accuracy_q_v1_fixed_si20 accuracy_q_v1_fixed_si20.json)
(cd iotlab/radio && python3 analyse_radio.py data/radio_m3a_serial.txt data/radio_m3a_21.oml m3 data/radio_m3a_serial.txt)
                                                             # radio_m3_summary.json, about 5 s
(cd iotlab/radio && python3 analyse_radio.py data/radio_nrfa_serial.txt data/radio_nrfa_5.oml nrf data/radio_nrfa_serial.txt \
  && cmp radio_nrf_summary.json data/radio_nrf_455113_summary.json && echo identical)   # first nRF52840 run
(cd iotlab/radio && python3 analyse_radio.py data/radio_nrfb_serial.txt data/radio_nrfb_5.oml nrf data/radio_nrfb_serial.txt)
                                                             # radio_nrf_summary.json (= data/radio_nrf_455117_summary.json)
(cd iotlab/radio && python3 radio_decisions.py radio_m3_summary.json radio_nrf_summary.json)   # radio_decisions.json
(cd midA && python3 verdict_rule_check.py)                   # verdict_rule_check.json; prints what verdict_rule_check.log holds
(cd recv && python3 verify_q.py)                             # about 75 s; prints runs 220 failed 0
(cd recv && python3 verify_q_v1.py)                          # about 70 s; prints runs 220 failed 0
python3 capsplat/caps_platforms.py                           # prints what capsplat/caps_platforms.log holds
python3 tolA/verify_ceiling.py                               # prints what tolA/verify_ceiling.log holds
python3 records/records.py                                   # records/*.json; prints what records/records.log holds
(cd iotlab && SOURCE_DATE_EPOCH=1791417600 python3 fig_power_trace.py)    # fig_power_trace.png and .pdf (Figure 2)
(cd blindsel && python3 -I blind_vs_caps.py)                 # blind_vs_caps.json (1 nJ per MAC over the radio sweep; the four platforms)
(cd blindsel && python3 -I blind_conf.py)                    # blind_conf.json (the twelve further datasets)
(cd blindsel/verify && python3 -I verify_blind.py)           # verify_out.json (independent recomputation)
(cd blindsel && python3 -I picks_conf.py)                    # picks_conf.json (compute-blind picks per further dataset)
(cd boot && python3 block_boot.py)                           # block_bootstrap.json; about 10 s
python3 records/lowest_energy_arm.py                         # records/lowest_energy_arm.json (Section 5.2, Figure S3 note)
python3 records/caps_single_scale.py                         # records/caps_single_scale.json (Section 5.1, integer input scale)
(cd netx && python3 gen.py && python3 gen2.py)              # app/netx_models.h, app/netx_models_q.h, app/netx_table.h, netx_host.json, netx_plan.json
(cd netx/check && bash build.sh)                             # out_*.txt, expected_decisions.json; prints mismatches: 0
                                                             # (needs gcc-arm-none-eabi and qemu-user; without them, python3 check.py checks the shipped outputs)
(cd netx/check/rx2 && gcc -O2 t.c -o t && ./t)               # prints 767683d4 e96af6ca: the sink v2 hash and check value of one frame, as analyse_netx_17b.py computes them
(cd netx && python3 analyse_netx.py m3_float data/netx_m3f_serial.txt data/netx_m3f_21.oml m3f)              # netx_m3f_summary.json (frozen rule, run 455253)
(cd netx && python3 analyse_netx_desc_drift.py m3_float data/netx_m3f_serial.txt data/netx_m3f_21.oml m3f)   # netx_m3f_DESCRIPTIVE_drift_summary.json
(cd netx && python3 analyse_netx_17a.py m3_int data/netx_m3i_serial.txt data/netx_m3i_21.oml m3i)            # netx_m3i_summary.json
(cd netx && python3 analyse_netx_17a.py m3_float data/netx_m3f2_serial.txt data/netx_m3f2_21.oml m3f2)       # netx_m3f2_summary.json
(cd netx && python3 analyse_netx_17b.py nrf data/netx_nrf4_serial.txt data/netx_nrf4_5.oml nrf4)             # netx_nrf4_summary.json
```
The `netx` summaries are written in `netx/` and equal the copies in `netx/data/`; `NETX_RESULTS.txt` collects their
outcomes. Where alignment was accepted, `blind_vs_caps` is the measured energy of compute-blind selection over that of
CAPS, minus 1, summed over the valid cells. `check/build.sh` compiles `app/netx_pred.h` with the firmware's flags and
newlib and runs it under `qemu-arm`; `check/rx2/t.c` is the C side of the sink v2 hash check.

In `clt_summary.json` the keys `ta_med`, `H2_ta`, `p_h2_ta` and `n_beatR_ta` hold the CLT values (names reused from
the TA analysis). `accuracy_q_v1_fixed_si20.json` was written with the first emulator (Deviations, item 9); to
regenerate it:

```bash
(cd iotlab/fixed && cp accuracy_q.json accuracy_q_v2.json && python3 run_with_v1_emulator.py accuracy_q.py \
  && mv accuracy_q.json accuracy_q_v1_check.json && mv accuracy_q_v2.json accuracy_q.json \
  && cmp accuracy_q_v1_check.json accuracy_q_v1_fixed_si20.json && echo identical)
```
Training scripts: `retrain_lw.py`, `ta_train.py`, `clt_train.py`, `conf_train.py <dataset> <model>`,
`confB/code/stb_run.py` (inside `confB/run_<deployment>`, `PYTHONPATH=../code`: `obs cheap`, then
`obs lightweight|cnn_lstm <first seed> <last seed plus one>`), `lb24_run.py <dataset> cheap|lightweight`,
`mid_run.py <dataset> mlp_mid|cnnlstm_small`; the `lane*.sh` and `finish.sh` schedulers show the order used.

## Firmware

Each group starts in `caps/`. Replace `<RIOT>` by a RIOT checkout at tag 2025.07 (commit
eb65305cf9f1b7affb50b17af5c12341b83a8636), or clone RIOT into `caps/iotlab/RIOT` and drop `RIOTBASE=<RIOT>`. The
first build downloads CMSIS into the RIOT tree, so it must be writable and online.

```bash
cd avrsim && python3 gen.py                                  # fw/models.h, ref.json
cd fw && avr-gcc -mmcu=atmega128 -Os -Wl,--undefined=_mmcu,--section-start=.mmcu=0x910000 \
         -I/usr/include/simavr/avr main.c -o fw.elf -lm      # fw/fw.elf
cd .. && simavr fw/fw.elf > raw_out.txt 2>&1                 # raw_out.txt, about 45 s

cd iotlab && python3 gen.py                                  # app/models.h, ref.json
cd app && make BOARD=iotlab-m3 RIOTBASE=<RIOT>               # bin/iotlab-m3/caps_energy.bin
make BOARD=nrf52840dk RIOTBASE=<RIOT>                        # bin/nrf52840dk/caps_energy.bin
cd ../hostchk && gcc -O2 -I. host.c -o host -lm && ./host > out.txt

cd iotlab/fixed && python3 run_with_v1_emulator.py gen_q.py  # app_q/models_q.h, ref_q.json (see Deviations, item 9)
cd app_q && make BOARD=iotlab-m3 RIOTBASE=<RIOT>             # bin/iotlab-m3/caps_fixed.bin
cd ../host && gcc -O2 -I. host.c -o host -lm && ./host > out.txt

cd iotlab/radio                                              # images of the radio runs (PREREG15, Amendment 15a)
make BOARD=iotlab-m3 ROLE=TX BINDIRBASE=$PWD/bin_TX RIOTBASE=<RIOT>     # bin_TX/iotlab-m3/caps_radio.bin
make BOARD=iotlab-m3 ROLE=RX BINDIRBASE=$PWD/bin_RX RIOTBASE=<RIOT>     # bin_RX/iotlab-m3/caps_radio.bin
make BOARD=nrf52840dk ROLE=TX BINDIRBASE=$PWD/bin_TX RIOTBASE=<RIOT>    # bin_TX/nrf52840dk/caps_radio.bin
make BOARD=nrf52840dk ROLE=RX BINDIRBASE=$PWD/bin_RX RIOTBASE=<RIOT>    # bin_RX/nrf52840dk/caps_radio.bin

cd netx/app                                                  # images of the end-to-end run (PREREG17, Amendment 17b)
make BOARD=iotlab-m3 PERIOD_CNN_MS=400 PASSES=3 BINDIRBASE=$PWD/bin_tx_m3_float RIOTBASE=<RIOT>             # netx_tx_m3_float
make BOARD=iotlab-m3 NETX_INT=1 PERIOD_CNN_MS=120 PASSES=3 BINDIRBASE=$PWD/bin_tx_m3_int RIOTBASE=<RIOT>    # netx_tx_m3_int
make BOARD=nrf52840dk PERIOD_CNN_MS=60 PASSES=5 BINDIRBASE=$PWD/bin_tx_nrf RIOTBASE=<RIOT>                  # netx_tx_nrf
make BOARD=iotlab-m3 ROLE=RX BINDIRBASE=$PWD/bin_rx_m3 RIOTBASE=<RIOT>                                      # netx_rx_m3
make BOARD=nrf52840dk ROLE=RX BINDIRBASE=$PWD/bin_rx_nrf RIOTBASE=<RIOT>                                    # netx_rx_nrf
cd ../app_rx2 && make BOARD=nrf52840dk BINDIRBASE=$PWD/bin_nrf RIOTBASE=<RIOT>                              # netx_rx2_nrf
```
Each image in `netx/firmware/` is `arm-none-eabi-objcopy -O binary` of the ELF the build writes
(`<BINDIRBASE>/<board>/netx.elf`, `netx_rx2.elf` for the sink v2).

Checked from a fresh copy: `fw/fw.elf`, `raw_out.txt`, both `out.txt` files and all flash images (three in
`iotlab/firmware/` and `iotlab/fixed/firmware/`, eight in `iotlab/radio/firmware/`, six in `netx/firmware/`) rebuild
byte for byte; the images of `iotlab/radio/firmware/v1/` come from `v1/main.c` with the same Makefile. ELF files are
not shipped because they record the build directory; their hashes are in CHANGES_v1.6.md,
`iotlab/radio/PREREG15.sha256`, `AMENDMENT_15a.sha256` and `netx/firmware/ELF_SHA256.txt`, and each image is
`objcopy -O binary` of its ELF. Build logs write the RIOT path as `<caps>/iotlab/RIOT`. `avrsim/SHA256SUMS` and `iotlab/SHA256SUMS` are the manifests of the time; the second gives
`PREREG12.txt` its hash as written and lists two ELF files that are not shipped.

Radio runs on the testbed (Amendment 15a): one experiment per run, with the transmitting node under the
profile of PREREG12, `iotlab-experiment submit` and `wait`, a power cycle of both nodes (`iotlab-node --stop`,
`--start`), `serial_aggregator | tee <file>` within the node's 60-s wait, and a copy of the consumption record.
The end-to-end runs followed `netx/RUN_COMMANDS.txt`, written before them, with the sink changed by Amendments 17b and 17c;
the nodes used are in `netx/RUNS_LOG.txt`, and the images it names are the Firmware builds above.

## Data

- `confA/data/`: six NAB streams (MIT licence), identical to the files at NAB commit ea702d75; check with
  `cd confA/data && sha256sum -c ../DATA.sha256`.
- `confB/data/`: four SensorScope archives (CC BY 4.0, G. Barrenetxea, "Sensorscope Data," Zenodo, 2019,
  doi:10.5281/zenodo.2654726), unmodified; check with `cd confB && sha256sum -c DATA.sha256`.
- Indoor data, not redistributed: `energydata_complete.csv` (github.com/LuisM78/Appliances-energy-prediction-data)
  (Candanedo et al., 2017) and `datatraining.txt`, `datatest.txt`, `datatest2.txt` (Candanedo and Feldheim, 2016;
  github.com/LuisM78/Occupancy-detection-data), also in
  the UCI repository. Put them in `confC/data/` and check with `cd confC/data && sha256sum -c ../DATA.sha256`.
- `iotlab/data/`, `iotlab/fixed/data/`: serial logs and power traces (time, record id, sample, timestamp s and µs,
  power W, voltage V, current A; one sample every 4.4 to 4.5 ms). `iotlab/records/`: the full consumption records of
  experiments 454471, 454508, 454553, 454554 and 454575 (check with `tar xzf iotlab_consumption_records.tar.gz &&
  sha256sum -c RECORDS.sha256` in `iotlab/records/`); `check_windows.py` shows each trace is a block of its record.
- `iotlab/radio/data/`: serial logs of both nodes and the full consumption record of the transmitting node for runs
  455109 (`radio_m3a_*`, M3), 455113 (`radio_nrfa_*`) and 455117 (`radio_nrfb_*`, nRF52840), and the failed run
  455108 (`failed_455108/`, not analysed); `DATA.sha256` gives the hashes logged before analysis.
- `netx/data/`: serial captures of both nodes and the sender's full consumption record (`.oml`) for runs 455253
  (`netx_m3f_*`, M3 floating point, descriptive), 455256 (`netx_m3i_*`, M3 integer), 455264 (`netx_m3f2_*`, M3
  floating point) and 455268 (`netx_nrf4_*`, nRF52840), with the summaries and analysis logs; `data/failed/`: serial
  excerpts of the three runs stopped as technical failures (455262, 455263, 455267; no power data read).

## Records written for versions 1.6 and 1.7

Numbers that had been printed or copied by hand are written to files by scripts that read only shipped outputs:
`records/records.py` (Tables 2 and S9, Section 5.2 totals, Section 5.1 numbers, the CNN-LSTM at 6,440 MACs),
`adaptA/gate1_build.py`, `iotlab/replication.py` (criterion in `iotlab/REPLICATION_CRITERION.txt`), `iotlab/fixed/run_with_v1_emulator.py`,
`iotlab/records/check_windows.py`, `cltA/recheck_clt.py`, `recv/verify_q*.py`, `midA/verdict_rule_check.py`,
`confB/SEGMENTS.sha256` and `iotlab/radio/data/DATA.sha256`. Logs of scripts re-run unchanged sit next to them, with
paths written as `<caps>`; some logs end with lines the schedulers appended (`ANALYSED`, `LANEDONE`). Version 1.7 adds
`blindsel/picks_conf.py` (compute-blind picks per dataset on the twelve further datasets), `records/lowest_energy_arm.py`
(the low-cost arm of lowest test-span energy as comparator) and `records/caps_single_scale.py` (CAPS on the M3 in
integer code with the firmware's single input scale).
`tolA/verify_ceiling.py` prints the Section 5.2 totals as first computed, which also count the TA model (−3.1%
and −3.7%); `records/totals_611.json` holds those the paper reports. The integer suppression used in Table 1 is that of `qemu.py`; the first emulator gives
the same decisions.

## Protocols

The analyses followed protocols written before they were run, except the screening budgets and totals of Section
5.2, CAPS per platform and the crossover radio cost of Section 5.1 (computed after the measurements, as the paper
states) and the records of versions 1.6 and 1.7. The comparison with compute-blind selection was computed after the
main results, with the selector fixed by `PREREG16.txt` before computing; its first draft (16:58, a different,
descriptive question) and that draft's script, never run, are in `blindsel/superseded/`. `PREREG4.txt` was written
after the test-span headroom, so its agreement with the test span is not confirmatory. Times are file times on the authors' machine
(Baghdad); each protocol file is older than every output it governs, and times typed inside protocols are 1 to 14
minutes later. Four protocols name their purpose instead of the question that prompted them, and `PREREG7.txt`
replaces a first name; no rule changed.

| File | Written | SHA-256 (shipped) | SHA-256 as written, if different |
|---|---|---|---|
| `adaptA/PREREG.txt` | 12 Oct 20:37 | `851db84e…` | |
| `adaptA/PREREG2.txt` | 12 Oct 20:52 | `fc0d3f1b…` | |
| `tolA/PREREG3.txt` | 12 Oct 23:11 | `bf1added…` | |
| `adaptA/PREREG4.txt` | 12 Oct 23:29 | `067ebf17…` | |
| `cltA/PREREG5.txt` | 13 Oct 08:46 | `3728c2e2…` | |
| `confA/PREREG6.txt` | 13 Oct 10:14 | `4b2f0e10…` | |
| `confB/PREREG7.txt` | 13 Oct 11:07 | `54eba788…` | `02c0fe62…` |
| `confB/AMENDMENT_7a.txt` | 13 Oct 11:11 | `23caa56f…` | |
| `shiftA/PREREG8.txt` | 13 Oct 11:23 | `938f3cff…` | |
| `lb24/PREREG9.txt` | 13 Oct 11:36 | `ff625339…` | `24854153…` |
| `confC/PREREG10.txt` | 13 Oct 13:03 | `f683e2f3…` | |
| `avrsim/PREREG11.txt` | 13 Oct 13:19 | `53a66636…` | `7a76fed4…` |
| `iotlab/PREREG12.txt` | 13 Oct 19:12 | `562fe6b2…` | `911ed965…` |
| `iotlab/fixed/PREREG13.txt` | 13 Oct 22:01 | `924e65c9…` | |
| `midA/PREREG14.txt` | 13 Oct 23:23 | `d1077e62…` | `fada554f…` |
| `iotlab/radio/PREREG15.txt` | 14 Oct 21:25 | `e6ef8b92…` | |
| `iotlab/radio/AMENDMENT_15a.txt` | 14 Oct 21:45 | `6673f1e3…` | `5f126ccf…` (first 20 lines, see Deviations, item 10) |
| `blindsel/PREREG16.txt` | 15 Oct 17:00 | `77b812f5…` | |
| `netx/PREREG17.txt` | 15 Oct 18:55 | `9595eeba…` | |
| `netx/AMENDMENT_17a.txt` | 15 Oct 19:58:03 | `a0c74752…` | |
| `netx/AMENDMENT_17b.txt` | 15 Oct 20:36 | `7e46c9cb…` | |
| `netx/AMENDMENT_17c.txt` | 15 Oct 21:40 | `75b6a4ec…` | |

The testbed's consumption records add a third-party time: each states when its recording started, and every one
is later than the protocol it followed (13 Oct: 454471 20:32, 454508 21:08, 454553 and 454554 21:40, 454575 22:30;
14 Oct: 455108 21:32, 455109 21:48, 455113 22:25, 455117 22:32; 15 Oct: 455253 18:59, 455256 19:58:36, 455264 20:32,
455268 21:41). The three end-to-end runs stopped as technical failures have no record.

## Deviations

Apart from two amendments, none changed a rule or threshold: Amendment 15a replaced the validity rule of PREREG15
(item 10) and Amendment 17a the alignment rule of PREREG17 (item 15).

1. 5.3 statistic: when `PREREG14.txt` (B) was written, persistence suppression on six of the thirteen datasets had been seen.
2. 5.2: `PREREG4.txt` was written after the test-span headroom; Gate 2 was not run because Gate 1 passed
   on 2 streams, not the 3 required.
3. 5.3 NAB: fastness had been computed on every NAB series, and two rows of one stream printed by a connectivity
   check (both stated in `PREREG6.txt`). Concurrent runs overwrote some records: `recover.py` recomputed every
   surviving record from its weights (all matched) and rebuilt the lost ones; the files before recovery are in
   `json_before_recovery/`. The first analysis stopped at its count check (`analyse_conf_failed_first.log`).
4. 5.3 SensorScope: Amendment 7a was written from the column definitions and archive listings only, before any station file was opened.
5. 5.3 indoor: the first analysis stopped at its own check (steps counted per node); fixed in one line, first
   version kept.
6. 4.4 ATmega128L: `PREREG11.txt` names the MICA2, whose 7.37 MHz clock the simulation uses; the Intel motes were
   MICA2DOT (same processor, 4 MHz).
7. 4.4 IoT-LAB: the first serial capture of the M3 run failed and was repeated before any analysis.
8. 4.4 alignment: clock offset by correlation, and a 45-sample running median (added for the noisy nRF52840 traces),
   used on every board's trace only to locate the phases (phase energies are means of the raw samples), added after the
   first runs (earlier scripts kept); a wider M3 window was taken from the same record (`m3_cons_cut2.txt`).
9. 5.1 integer code: `PREREG13.txt` followed the floating-point runs; the first firmware overflowed its stack
   (`main_v1_stackoverflow.c`, not measured); the first emulator (`qemu_v1_si20.py`) rounds iio_networkin inputs to
   zero, `qemu.py` scales per stream, a choice made after that result; both give the same decisions. The firmware was
   generated with the first emulator, hence `run_with_v1_emulator.py`.
10. 4.4 radio: run 455108 failed (late serial capture; after a reset the radio driver did not start; the link-layer
    counters of the validity rule are not updated by this RIOT version). `AMENDMENT_15a.txt`, written before any of its
    power data were analysed, changed the firmware and procedure and replaced the validity rule by the receiving
    node's counts (at least 99% received); the run was then repeated. Item 6 (power cycle) was added 17 s after the
    amendment's hash list, before the repeat, so the list gives the first 20 lines (`5f126ccf…`). `sha256sum -c` on
    `PREREG15.sha256` and `AMENDMENT_15a.sha256` reports the files changed since each list (earlier versions in `v1/`)
    and the ELF files as missing.
11. 4.4 radio: the amended `analyse_radio.py` stopped on run 455109 before printing anything (the record's header
    lines also have eight fields); one line was changed to skip them; both frozen versions are in `v1/`.
12. 5.1 radio: the nRF52840 alignment reached 0.73 against the required 0.8, so run 455113 counted as a technical
    failure and was repeated once (455117), as the protocol allows; the repeat also reached 0.73, the two agree within
    1%, and the paper reports these values as descriptive. The M3 run passed (0.835).
13. 5.5: one analysis import stopped on a key error before printing anything.
14. 5.5: `PREREG14.txt` asks for verdicts "as in Section 6.3" (now 5.2) but describes one-sided tests with Holm's correction,
    which `analyse_mid.py` applies and Table S10 reports; Section 5.2 uses two-sided tests against 0.05/22. Under that
    rule the small CNN-LSTM saves on 2 streams instead of 3, and the conclusion holds under both
    (`verdict_rule_check.json`).
15. 4.4 end to end: run 455253 (M3, floating point) failed the PREREG17 alignment rule by clock drift (0.12 s against
    0.1 s). Amendment 17a (offset interpolated between the two SYNC fits, drift at most 0.5 s), written before the other
    runs, applies to them; the M3 floating-point run was repeated (455264) and the first is descriptive (its energies
    agree with the repeat within 0.5%). Its descriptive re-analysis with drift correction (`analyse_netx_desc_drift.py`)
    was run 3 minutes before the amendment was written.
16. 4.4 end to end: the serial output of the nRF52840 sinks at saclay was corrupted on two nodes and with two sink
    firmwares (455262, 455263, 455267, stopped as technical failures, no power data read): Amendment 17b (sink v2:
    cumulative counts and hashes), then Amendment 17c (an M3 sink, m3-1). `PREREG17.txt` names nRF52840 node 6 as the
    sink.
17. 4.4 end to end: frames are 7 bytes against the 4-byte report of PREREG15, whose unicast energy the prediction uses
    (stated in `PREREG17.txt`).
18. 4.4 end to end: `sim_netx.py`, the test tool, gained the options `SIM_DRIFT`, `SIM_SINK=v2` and `SIM_CORRUPT` after
    the freeze; its hashes are in the amendment lists. `sha256sum -c` on `MANIFEST.sha256` and
    `AMENDMENT_17a.sha256` reports it changed, and on `MANIFEST.sha256` and `AMENDMENT_17b.sha256` the ELF files as
    missing.
19. 4.4 idle power: `analyse_iotlab.py` and `fixed/analyse_fixed.py` take the median of the idle-phase means, where
    `PREREG12.txt` says the mean; with the mean, energies change by at most 0.25% and no decision changes.
20. 5.3 SensorScope: `confB/prep_deploy.py` updates the run length before the temperature check, the order that
    Amendment 1 of Grand St Bernard corrected; the four archives have no out-of-range or clock-reset rows, and the
    corrected order gives byte-identical preparations (`confB/SEGMENTS.sha256`).
