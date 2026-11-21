# Version 1.8

Accompanies the shortened manuscript, which no longer has a separate Supplementary material and adds batched
reporting (Sections 3.7 and 5.7) and the NLMS dual predictor of [6] to Table 5. Old and new numbers are in
`NUMBERING_v1.8.md`. `SHA256SUMS_v1.8` lists every file added or changed.

## Added

- `caps/batch/`: the batching analysis (`RULE5.txt`, `batch.py`, `heldout.py`, `c3.py`, `table7_rows.py`,
  `oracle_lib.py` and their outputs) and the node run `iotlab/` (PREREG16, Amendments 16a and 16b, application,
  four flash images, serial logs, consumption records, analyses and summaries). See `caps/batch/README.md`.
- `caps/nlms/`: `RULE6.txt`, `nlms_zoh.py`, `nlms_results.json` and their hashes. See `caps/nlms/README.md`.
- `CHANGES_v1.8.md`, `NUMBERING_v1.8.md`, `SHA256SUMS_v1.8`.

## Edited after freezing

Two frozen scripts named the working folder of the original run by its absolute path. In this copy those lines
read the package folders instead; no other line changed, and the outputs rebuild byte for byte.

| File | Frozen hash (record) | Hash in this version |
|---|---|---|
| `caps/batch/iotlab/gen_sched.py` (lines 4 and 6) | `6c5febb6…` (`PREREG16.sha256`) | `93a9e3f0…` |
| `caps/nlms/nlms_zoh.py` (line 5) | `110d031a…` (`RESULTS6.sha256`) | `527a1f5b…` |

`batch.py`, `heldout.py` and `c3.py` (not hashed when run) had the same edit, and `oracle_lib.py` takes its run
folder from `CAW_RUN`. The flash images are shipped as binaries, as in version 1.7, because the ELF files record the
build folder; `caps/batch/iotlab/firmware/ELF_SHA256.txt` repeats the ELF hashes of `PREREG16.sha256`.

## Checks

From a fresh copy set up as in `README.md`: `batch.py`, `heldout.py`, `c3.py`, `table7_rows.py`, `gen_sched.py`
(`sched.json`, `app/replay_tab.h`) and `nlms_zoh.py` rebuild their outputs byte for byte, and `analyse_batch.py`
reproduces both node-run summaries exactly. The four flash images rebuild byte for byte from `caps/batch/iotlab/app/`.

## Changed

- `README.md`: version, layout, numbering note, reference concordance (column for this version), citation, co-author name as on his Scopus profile (Mustafa Musa Jaber).
- `TABLES_AND_FIGURES.md`: a section for the tables added in this version.
- `.gitignore`: the firmware build folders of `caps/batch/iotlab/app`.
