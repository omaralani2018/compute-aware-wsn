# Version 1.7

Accompanies the reorganised manuscript and its Supplementary material (old and new numbers in `NUMBERING_v1.7.md`) and adds `caps/blindsel/`
(compute-blind selection against CAPS, Sections 4.5 and 5.1 to 5.3) and `caps/netx/` (dual prediction run end to end
on IoT-LAB nodes over their own radios, Sections 4.4 and 5.1); see `caps/README.md`. `SHA256SUMS_v1.7` lists every
file added or changed.

## Added

- `caps/blindsel/`: `PREREG16.txt` and its hash, `MANIFEST.sha256`, `blind_vs_caps.py` (1 nJ per MAC over the radio
  sweep; the four platforms) with `blind_vs_caps.json`, `blind_conf.py` (the twelve further datasets) with
  `blind_conf.json`, `verify/verify_blind.py` with `verify/verify_out.json` (independent recomputation), the record
  `picks_conf.py` with `picks_conf.json` (compute-blind picks per further dataset), and `superseded/`: the first draft of
  PREREG16 (16:58) and its script, replaced at 17:00 before anything was computed (no output exists).
- `caps/netx/`: `PREREG17.txt`, `AMENDMENT_17a.txt` to `AMENDMENT_17c.txt` and their hash lists, the freeze manifest
  `MANIFEST.sha256`, generators `gen.py` and `gen2.py`, RIOT applications `app/` (sender and sink) and `app_rx2/`
  (sink v2) with build logs, host check `check/`, the analyses, `sim_netx.py`, six flash images in `firmware/` with
  their `THIRD_PARTY_NOTICES.txt`, serial
  captures, consumption records and summaries in `data/`, `RUN_COMMANDS.txt`, `RUNS_LOG.txt`, `NETX_RESULTS.txt`.
- `caps/boot/`: `block_boot.py` and `block_bootstrap.json`, the moving-block bootstrap over test rounds reported in
  Section 5.2, computed after the results (blocks of 48 and 96 rounds, 10,000 replicates, seed 20261017).
- `caps/records/lowest_energy_arm.py` and `caps/records/caps_single_scale.py` with their outputs, records of two
  statements of Sections 5.1 and 5.2 that had no output file.
- `CHANGES_v1.7.md`, `NUMBERING_v1.7.md`, `SHA256SUMS_v1.7`.

## Changed

- `README.md`, `TABLES_AND_FIGURES.md` and `caps/README.md`: numbers of the manuscript this version accompanies, the
  two new folders, a column for version 1.6 docs in the reference concordance, and the creator of the Sensorscope data
  [43] (G. Barrenetxea, as in the Zenodo record), also in `LICENSE`; `.gitignore` (the netx build folders); `caps/setup.sh` (the `nab_repo` link for `boot/`); in `README.md` and `caps/README.md`, the setup
  copy step, the files that need the raw Intel archive or the version 1.5 records, and Deviations 19 and 20;
  `SHA256SUMS_all`. No other file of version 1.6 changed.

## Differences from the files as run

Files in `caps/blindsel/` and `caps/netx/` are copied unchanged except those below and the record `picks_conf.py`
written for this version. `analyse_netx_desc_drift.py` was run from `netx/data/` and is shipped, unchanged, in `netx/`.

Two files named the run directory by its absolute path; no other line changed.

| File | Change | SHA-256 as run or as written |
|---|---|---|
| `caps/blindsel/verify/verify_blind.py` | `SC` is now the `caps/` folder found from the script's location | `74e6b53eeab132d02f9498a1ea724c809824074fa2e0518a4ebee7c5c42f4ed8` |
| `caps/blindsel/PREREG16.sha256` | lists the bare name `PREREG16.txt` (same hash) | `8a95753c69029139bf13438af5d8f95ba9baef25955ce3e839525cc3d0204364` |

Six build logs: the absolute run directory is written as `<caps>`; no other character changed.

| File | SHA-256 as run |
|---|---|
| `caps/netx/app/build_rx_m3.log` | `7b0f843daf645df6d3ad4e14bf697b2ad8cc93eee4d66009760fcd5e25585046` |
| `caps/netx/app/build_rx_nrf.log` | `2cf010d09b62699bef7df4d553fb1a533d8203404e1fcd613c5caa7e2a4b3f05` |
| `caps/netx/app/build_tx_m3_float.log` | `58416445c01c2b66c018c8f1d3af6559be0d4b725da91e480813ca2356566ff4` |
| `caps/netx/app/build_tx_m3_int.log` | `358439f83d38fe473fd2039869c9c7590e2505d5c2ed2366f66bfd6dce5ffc9f` |
| `caps/netx/app/build_tx_nrf.log` | `3dd8d5a01687148d0e530130e6882fa47d13e7762bcd3ebd07c39b105e56a6ef` |
| `caps/netx/app_rx2/build_nrf.log` | `8955896478c817b9922403cf5f1f33c6516c041a78685c093aaf327dc16e9720` |

## Not included

- ELF files, which record the build directory; their hashes are in `caps/netx/firmware/ELF_SHA256.txt` (also in
  `MANIFEST.sha256` and `AMENDMENT_17b.sha256`, under `fw/`), and each shipped image is `objcopy -O binary` of its ELF.
- Build objects, host-check binaries, the RIOT tree, a draft of PREREG17 written before the freeze, logs of sender
  builds made before the final ones and of an unused M3 build of the sink v2, captured compiler flags, console logs of
  `gen2.py` and of a package install, the summaries of the synthetic-trace tests of the analyses (`sim_netx.py`), and
  a duplicate copy of `netx_m3f_summary.json`.

## Checks

From a fresh copy after `caps/setup.sh`: `blind_vs_caps.json`, `blind_conf.json`, `picks_conf.json` and
`verify/verify_out.json` regenerate byte for byte; `gen.py` and `gen2.py` regenerate the three headers,
`netx_host.json` and `netx_plan.json`; `check/build.sh` regenerates `out_*.txt` and `expected_decisions.json` and prints `mismatches: 0`; the five analyses
regenerate the shipped summaries; all six images in `netx/firmware/` rebuild byte for byte and equal `objcopy -O binary`
of the ELF files whose hashes are listed; the protocol files match their hash lists; every testbed recording of the
end-to-end runs started after the protocol it followed (`caps/README.md`, Protocols); no file holds an absolute path of
the machine the runs were made on.
