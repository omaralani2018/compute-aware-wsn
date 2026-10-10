# Changes in version 1.2 (2 October 2026)

Version 1.2 adds the matched comparison with three published dual-prediction schemes behind
Table 10 and the Section 6.8 paragraph of the paper: LMS dual prediction [13], AM-DR [25] and
LSTM-DPS [18] (reference numbers of an earlier draft; in the current manuscript they are [10],
[24] and [25]). `README.md` is updated to describe the additions and to correct one
paragraph, and `requirements.txt` gains two
lines (hyperopt and h5py). No other existing file is modified, and no result of versions 1.0 or
1.1 changes. Details are in `MATCHED_BASELINES.md`.

Added
- `src/ports_faithful.py`: line-by-line Python ports of `baseline/dual_prediction.m` and
  `myapproach_AMDR/AMDR_dual_prediction.m` from github.com/YasminFathy/AMDRIoT.
- `src/matched_baselines.py`: LMS and AM-DR under this protocol with the released and the
  single-error transmission rule; `results/matched_<stream>.json` for the eleven streams.
- `src/lstmdps_port.py`, `src/lstmdps_train.py`, `src/lstmdps_protocol.py`: LSTM-DPS from
  github.com/Ahmed-Fathalla/LSTM-DPS-for-IoT; `results/lstmdps_intel_runs.jsonl`, 150 runs
  (30 motes × seeds 0–4).
- `src/validate_ports.py`: `results/port_validation.json` and `results/lstmdps_port_validation.csv`.
- `src/summarise_matched.py`: `results/matched_summary.json` and `results/lstmdps_summary.json`.
- `results/matched_earlier_grids/`: the two earlier runs with narrower grids, with a README.
- `scripts/matched_runall.sh`, `scripts/run_lstmdps.sh`, `scripts/lstmdps_worker_loop.sh`: the
  schedulers used for the runs. The LSTM-DPS runs were split over three workers, restarted by
  `lstmdps_worker_loop.sh` after interruptions (the protocol script resumes from its output file),
  and the three output files were merged and sorted by seed and mote into
  `lstmdps_intel_runs.jsonl`.
- `MATCHED_BASELINES.md`, `CHANGES_v1.2.md`, `SHA256SUMS_v1.2`.

Checks
- `validate_ports.py`, on the authors' data selection at e_max = 0.5 °C: the LMS and AM-DR ports
  give 7.4% and 4.3% transmitted on mote 11 (published 8% and 5%) and 17.8% and 13.4% on mote 30
  at the Fig. 9 settings (published 18% and about 13%); the Fig. 8 value for mote 11 (about 7%) is
  not reproduced (3.5%). AM-DR with the published α transmits on the same rounds as with α = 0 on
  all five motes.
- Harness equivalence: on the Intel test slice, 30 motes, τ = 0.10 °C, released rule, the
  predictors of `matched_baselines.py` and the line-by-line ports with the same 12-row warm-up
  make the same decision in all 226,170 rounds, for the published and the validation-selected
  settings of both schemes.
- LSTM-DPS replay with the released `_refit.h5` models: the transmission count is exact in all
  24 released experiments on motes 1 and 11; motes 30 and 49 have no readings in the released
  data file, and mote 13 has a different number of readings from the saved runs.
  On motes 1 and 11 at 0.10 °C, LSTM-DPS suppresses 94.0% and 95.0% and persistence, on the same
  series, 92.7% and 94.2%.
- Every LMS, AM-DR and LSTM-DPS test run passes the independent-receiver check; the largest sink
  error over the 44 LMS and AM-DR test runs is 0.99999 τ.
- From a clean unpack of this package on Linux, `matched_baselines.py` regenerates the eleven
  `matched_*.json` byte for byte, `summarise_matched.py` regenerates both summaries byte for byte,
  `validate_ports.py` regenerates `port_validation.json` and `lstmdps_port_validation.csv` byte for
  byte, and a rerun of LSTM-DPS on mote 0, seed 0 reproduces its line of
  `lstmdps_intel_runs.jsonl` in every field except the wall-clock `seconds`.
- A second clean rerun on Windows 11 (Python 3.11.9, same package versions) reproduced the same
  files identically apart from line endings, except four values of `max_sink_err_LMS` in
  `port_validation.json`, which differ from the twelfth significant digit on (the line-by-line LMS
  port uses `np.dot`, which goes through the platform's BLAS; see `MATCHED_BASELINES.md`). The
  LSTM-DPS retraining was not rerun on Windows.
- The numbers of Table 10, of the Section 6.8 paragraph and of the abstract match
  `matched_summary.json` and `lstmdps_summary.json`.

Run order for the added files (continuing the numbering of `CHANGES_v1.1.md`; inside `run/`,
which also needs `results/lstmdps_intel_runs.jsonl`)

16. `python3 matched_baselines.py <stream>`  for each of the eleven streams -> `matched_<stream>.json`
17. `python3 lstmdps_protocol.py 0,1,2,3,4 $(seq -s, 0 29) lstmdps_rerun.jsonl`
    -> LSTM-DPS runs (about 5.4 CPU-hours; resumes if interrupted). `summarise_matched.py` reads
    the shipped `lstmdps_intel_runs.jsonl`; to summarise a rerun, rename its output to that name
18. `python3 summarise_matched.py`            -> `matched_summary.json`, `lstmdps_summary.json`
19. `python3 validate_ports.py --lstmdps PATH` -> `port_validation.json`, `lstmdps_port_validation.csv`
    (needs `raw/data.txt` and a clone of LSTM-DPS-for-IoT with `data/Intel-DS.csv` extracted)
