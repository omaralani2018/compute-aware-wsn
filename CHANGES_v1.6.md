# Version 1.6

Adds `caps/` (Sections 6.6 test, 6.11 to 6.15 and 7.1): protocols, code as run, per-seed results, weights, logs,
firmware and the processor and radio measurements; see `caps/README.md`. `SHA256SUMS_v1.6` lists every file added
or changed.

## Added

- `caps/` and its `README.md`, including 1,780 saved models and the radio measurement `caps/iotlab/radio/`.
- `CHANGES_v1.6.md`, `SHA256SUMS_v1.6`.

## Changed

- `README.md` (rewritten shorter), `TABLES_AND_FIGURES.md` (Tables 12 to 16, Sections 6.6, 6.11 to
  6.15 and 7.1), `LICENSE` (outputs under `caps/`, third-party parts, SensorScope creator LCAV, EPFL),
  `SHA256SUMS_all`, `.gitignore`, and one word of `RUN_ORDER.md` ("low-cost arms"). No other file of version 1.5
  changed.

## Differences from the files as run

Files in `caps/` are copied unchanged except those below (hashes of the files as run or as written) and the records
written for version 1.6 (`caps/README.md`, Records).

Five protocols: purpose lines now state the purpose, and `PREREG7.txt` replaces a first name; no rule changed.

| File | SHA-256 as written |
|---|---|
| `caps/avrsim/PREREG11.txt` | `7a76fed465ec2cd3c88997e06c122d7240c7a562a541ae709566fbaff7423a7b` |
| `caps/confB/PREREG7.txt` | `02c0fe6222eab8507e09bd6042b095eef4e88c489f4513b2bf2d2dfd5e6296a2` |
| `caps/iotlab/PREREG12.txt` | `911ed9650505b2f3502baec030842bebc5e368d397babcf820ef3ee245e4f566` |
| `caps/lb24/PREREG9.txt` | `2485415366ec485160b17c3e90b447e02d7e9eae83ab1aec11f61dc6d76b03cd` |
| `caps/midA/PREREG14.txt` | `fada554f164145e782e54ea6e05458adab339d5151663f30b70b96994f46fa45` |

`caps/iotlab/radio/AMENDMENT_15a.txt`: item 6 was appended 17 s after its hash list, which gives the first 20 lines
(`5f126ccf944ef8ec94e88194f81aea344ef6b0eb1f1da8345206ece8fcc00efb`).

Nine schedulers: the absolute run directory is replaced by the script's directory; no other line changed.

| File | SHA-256 as run |
|---|---|
| `caps/confB/finish.sh` | `5292f94148c3679695bb1b3aaafed7b709140337788536a06436ef39224de04f` |
| `caps/confB/lanes.sh` | `c11eb62c129d59fc4f46f11295072ec6c84c1c89032053f008e7fc4abc58bb93` |
| `caps/confC/finish.sh` | `b49086f4df47a5771e112be37ad29c2db019e9a1ffaa7e4e46df1eba2dd39345` |
| `caps/confC/lanes.sh` | `322ed6b112b2a1b28af61ce1d3ea850a0fcb3ead728fe9020ccd8c15bbbd86de` |
| `caps/lb24/finish.sh` | `4b46d3796e59f2e49ae9040e777fdb906641ae0e82456d18c2367585e55c54e1` |
| `caps/lb24/lanes.sh` | `51b11755597b102441c8c0f03742f768c0d77f576281e2882576bdf06ea1df11` |
| `caps/midA/finish.sh` | `7d3fadcb3cca821d2e8fa6bd31933877907994d4e7139b1f9310cdacff296464` |
| `caps/midA/lanes.sh` | `7aa11848eaea1f796e36d87703ee08b7f6bfe4ced035a68db37086768c70b302` |
| `caps/midA/lanes2.sh` | `e545f5743ac1d6e2294b84f9dd6b706603405dbe36b2d710baea9ddc21169158` |

Twenty-two logs: the absolute run directory is written as `<caps>`; no other character changed.

| File | SHA-256 as run |
|---|---|
| `caps/adaptA/lane3.log` | `91b27a825d3b53cd39f6aeed3c464d89c071d6706a60718b463b987bcac4598d` |
| `caps/adaptA/oracle_exchange_cpc.log` | `0055082d8637ea23015b50a2f3812970729121463472883c01dbc2a454dc150d` |
| `caps/adaptA/oracle_grok_asg_anomaly.log` | `b35437b8c99d717dd48e87ee0f69965d59315e9cc075b83f18199cc4ca5de9e0` |
| `caps/adaptA/oracle_iio_networkin.log` | `07df70e433b0ccf3ba39c501bb896115d505289cccd215f3817e590f0ff75b54` |
| `caps/adaptA/oracle_machine_temp.log` | `ec733245d6cfa9ee2ba7addf9dae1f312c1909b96a096f5883415e8cd1a096c4` |
| `caps/adaptA/oracle_occupancy.log` | `701f445b18a50d3ea923337ce7015d4f4d462bf5663779444e8287529936c112` |
| `caps/adaptA/oracle_rds_cpu_util.log` | `7c6c0c0fdee4b784b8a79df8926cfe25b8ff23a7f7a21e277a70d52cdd7dfd6e` |
| `caps/adaptA/oracle_twitter_aapl.log` | `70a85993aa304e14bc51d87709dc4bd6a32b3875312351ef60462dae7ba13cda` |
| `caps/adaptA/val_headroom.log` | `f9f79aa297e03e56919cfc6308ab0ca16d98ced980450b67982db1d83e0454ea` |
| `caps/confA/analyse_conf_failed_first.log` | `069a83baea227c0351c70f1ebc88f47e34cd392d7e34528c1bcd00e8ef384ce9` |
| `caps/confB/analyse_confB.log` | `92264c858cfe3c430836ade513d1557bf4a69b7a490672377cc06c0a94ff94c5` |
| `caps/confC/analyse_conf.log` | `80e5eea5861d66653f1fec8f6b3e87bf1bdb451a5d34babe0cf302c9e7f28b39` |
| `caps/confC/analyse_conf_failed_first.log` | `4cbd9bf8594bdd3bd18a2c0263079d7e1e9f388c561f85665d3c9793e3a18056` |
| `caps/iotlab/app/build_iotlab-m3.log` | `f606b3399bded78a9160eff0e8d5eba64c3294c968f0aa5146201647856fd634` |
| `caps/iotlab/app/build_nrf52840dk.log` | `de8471534903c54dcb6aab8d2a21312c5ba4bdcb16a1561e41fec4ef1ddca61d` |
| `caps/iotlab/fixed/accuracy_q.log` | `02a105de9982e88ae2b5389a4aa7d6509b57e995aaa608209118e382b67b7bce` |
| `caps/iotlab/fixed/accuracy_q_v1_fixed_si20.log` | `e920a631e57be47d563ff4ca6be9595ece78377207250bebd00ca1786d5b0f7c` |
| `caps/iotlab/fixed/app_q/build.log` | `a4f2b87592958ebcacf053c4a4279904eec0761b9106115b081a732aeb43e349` |
| `caps/iotlab/radio/build_iotlab-m3_RX.log` | `4528c9d75a438d96ddff11e5ef4bfe150a421fb10f802acb317372ef428c9988` |
| `caps/iotlab/radio/build_iotlab-m3_TX.log` | `9a6cf72410c34c930b20f8f6fde5614f0985049186da8dc97d4e49ecc97cc70b` |
| `caps/iotlab/radio/build_nrf52840dk_RX.log` | `ebbd9b9a4469239dbfc53f3c876355469fafe9f8559feda21b39f335a18c902d` |
| `caps/iotlab/radio/build_nrf52840dk_TX.log` | `22e7972afa0e56fac8d540195e95037d6f3edad4fa7ff3eb2d384df0acdd7c99` |

`caps/iotlab/iotlab_m3_summary.json` is written by the shipped `analyse_iotlab.py`; the file of the run
(`d04b95338da1a4a5a27ee9e49fb5727d5712af219e8f039cea74c3f2215a0475`) differs only in the `round` label of the idle
phases.

## Not included

- ELF files, which record the build directory: the node builds (`caps_energy_m3.elf` `ebd206da9aa528f2f72bd8fb04e458f2aeea2e7ff0c79c432616cc0844970e82`,
  `caps_energy_nrf52840dk.elf` `c4e0d2bdff60d3459c00482823ebaa5ad9dcc427ae31ffd3d011b66ac6348015`, `caps_fixed_m3.elf`
  `ca61b24e7f658f88cf8337949fdf855385de2b342e0b4713af5cfd5d2b8a96c3`, and the first, stack-overflowing build of the integer firmware, `ed84cacb9c047ca2978f41ff6ebd11f63f851dc817aab48d10df902a8cef00a2`) and the eight radio builds
  (hashes in `caps/iotlab/radio/PREREG15.sha256` and `AMENDMENT_15a.sha256`); the shipped images are `objcopy -O
  binary` of them.
- The RIOT tree (tag 2025.07, unmodified), build objects, host-check binaries, the indoor data, the SensorScope
  segment files that `setup.sh` regenerates, the first written radio procedure (superseded by Amendment 15a, item 6),
  duplicate copies, a toolchain smoke test, a diagnostic script, links and Python caches.

## Checks

From a fresh copy after `caps/setup.sh`: every analysis summary and generated header regenerates byte for byte, and
so do the analysis logs apart from `<caps>` paths, scheduler-appended lines and the two provenance lines of the
radio decision log; every firmware image rebuilds byte for byte; the SensorScope
segments match `confB/SEGMENTS.sha256` and the NAB files their pinned commit; the receiver check passes on all
2 × 220 integer-code runs; Tables 12 and 13 match `records/` cell by cell; every power trace is a block of its full
record; an independent second run of every command from the documentation alone reproduced every output; no file
holds an absolute path of the machine the runs were made on.
