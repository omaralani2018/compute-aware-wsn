# Batched reporting (Sections 3.7, 4.4 and 5.7; Tables 7 and 8)

Added in version 1.8. Numbers are those of the current manuscript.

## Analysis (this folder)

`RULE5.txt` is the plan, written before any batching number was computed. The frame cost is the straight line
through each node type's measured 4-byte and 100-byte broadcasts (`caps/iotlab/radio/`).

| File | Output | Paper |
|---|---|---|
| `batch.py` | `dev_batch.json`: S0, S1 and S2 at D = 5, 10, 25 on the eleven development streams | Table 7, Section 5.7 |
| `heldout.py` | `heldout_batch.json`: the same on the eight datasets not used in development | Table 7, Section 5.7 |
| `c3.py` | `c3.json`: net saving of the lightweight model under S0 and S2 with each seed's own transmit decisions | Section 5.7 |
| `table7_rows.py` | `table7_rows.json`: the rows of Table 7 | Table 7 |

`oracle_lib.py` is the shared loader (reference arm, closed-loop prediction). The scripts read the `run/` folder
of the top-level `README.md` (or the folder in `CAW_RUN`). `c3.py` also needs the lightweight weights:
`cp caps/adaptA/weights/*.npz run/weights/`. `heldout.py` needs the indoor files in `caps/confC/data/`
(see the README there). With these in place all four outputs rebuild byte for byte.

## Node run (`iotlab/`)

`PREREG16.txt` (protocol, written before flashing; its hashes in `PREREG16.sha256`), `AMENDMENT16a.txt`
(alignment window, after the M3 run) and `AMENDMENT16b.txt` (nRF52840 capture and validity, before its trace was
analysed). This PREREG16 is not the one in `caps/blindsel/`, which belongs to version 1.7.

| Path | Contents |
|---|---|
| `app/`, `gen_sched.py`, `sched.json` | RIOT application and the replayed schedules (taxi, traffic, Intel mote; D = 25) |
| `firmware/` | the four flash images (`objcopy -O binary` of the ELF files whose hashes are in `PREREG16.sha256`; see `ELF_SHA256.txt`, `BIN_SHA256.txt`, `THIRD_PARTY_NOTICES.txt`) |
| `run_m3/`, `run_nrf/` | serial logs, consumption records and summaries of runs 458822 (M3, Grenoble) and 458828 (nRF52840, Saclay) |
| `analyse_batch_v0.py`, `run_m3/analyse_batch.py`, `analyse_batch.py` | the analysis as frozen, after Amendment 16a, and after Amendment 16b |
| `synth_test.py`, `RUN_COMMANDS16.txt` | check of the analysis on synthetic traces; the commands used on the testbed |

Rebuild the images (RIOT at commit eb65305c, arm-none-eabi-gcc 13.2.1; from `iotlab/app/`):

```bash
make BOARD=iotlab-m3 ROLE=TX BINDIRBASE=$PWD/bin_TX RIOTBASE=<RIOT>     # bin_TX/iotlab-m3/caps_batch.bin  = b16_tx_m3.bin
make BOARD=iotlab-m3 ROLE=RX BINDIRBASE=$PWD/bin_RX RIOTBASE=<RIOT>     # = b16_rx_m3.bin
make BOARD=nrf52840dk ROLE=TX BINDIRBASE=$PWD/bin_TX RIOTBASE=<RIOT>    # = b16_tx_nrf.bin
make BOARD=nrf52840dk ROLE=RX BINDIRBASE=$PWD/bin_RX RIOTBASE=<RIOT>    # = b16_rx_nrf.bin
```
All four rebuild byte for byte.

Rerun (from `iotlab/`):

```bash
python3 analyse_batch.py run_m3/b16_m3_serial.txt run_m3/b16_m3_21.oml m3 sched.json ../../iotlab/radio/radio_m3_summary.json
python3 analyse_batch.py run_nrf/b16_nrf2_tx.txt run_nrf/b16_nrf_5.oml nrf sched.json ../../iotlab/radio/radio_nrf_summary.json
```
Both summaries equal the shipped ones in `run_m3/` and `run_nrf/`.
