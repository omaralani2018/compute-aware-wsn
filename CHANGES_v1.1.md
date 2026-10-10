# Changes in version 1.1 (1 October 2026)

Version 1.0 regenerated four of the paper's nine figures and did not contain the routing simulation
behind Table 5. Version 1.1 adds the missing code and outputs. `README.md` is updated to describe the
additions; no other existing file is modified, and no result of version 1.0 changes.

Added
- `src/figures_extra.py`: Figures 2, 6, 7 and 9, plus `results/checkpoint_stats.json` (Section 6.9).
- `src/figures_pdf.py`: runs `figures.py` unmodified and also saves vector PDF.
- `src/routing_baselines.py`: routing and clustering baselines (Section 6.5, Table 5), extracted from the
  original analysis notebook and made standalone.
- `inputs/mote_locs.txt`: mote coordinates from the Intel Lab release (used only by the routing baselines).
- `inputs/intel_time_rows.csv`: start time of every row of `intel_time.npy` (written by `prepare_intel.py`).
- `inputs/old33_motes.json`: the 33-mote layout of earlier drafts, for the reproduction check.
- `src/text_checks.py` and `results/text_checks.json`: numbers quoted in the text (anchor radio cost and its
  alternatives, the band over which no verdict changes, open-loop fraction on the Intel test span, the
  no-interpolation stress test, which was previously recorded as computed by hand).
- `src/keras_equivalence.py` and `results/keras_equivalence.json`: NumPy-vs-Keras check on all saved models.
- `src/receiver_checks.py` and `results/receiver_checks.json`: independent-receiver check of the cheap arms with their
  validation-selected parameters at every tolerance (the flags stored in version 1.0 cover the learned models, the
  tolerance sweep, and the cheap arms only at 0.10 and, on the NAB streams, only in their untuned form).
- `src/epoch_counter_check.py` and `results/epoch_counter.json`: the evidence for indexing the Intel matrix by
  timestamp (Section 5.2). The docstring of `prepare_intel.py` (unchanged, as run) describes the counter as
  restarting "40-190" times per mote; that range counts every backward step of the counter in timestamp
  order in the valid readings, most of them small; backward jumps of more than 1,000 counts number 1 to 5 per mote.
- `src/interpolation_check.py` and `results/interpolation_check.json`: what the interpolation step R5 does to the
  Intel matrix (Section 5.2). The docstring of `prepare_intel.py` (unchanged, as run) states R5 as "linear
  interpolation of interior gaps of at most 8 consecutive bins"; the code, `interpolate(limit=8,
  limit_area="inside")`, fills the first eight missing bins of every interior gap, whatever its length, and
  the rest of a longer gap stays missing. The manuscript describes the code.
- `results/routing_30motes.json`, `results/routing_33motes.json`, `results/checkpoint_stats.json`.
- `figures/`: all data figures as PNG and vector PDF.
- `requirements.txt`: the package versions with which these checks reproduced the shipped outputs byte for byte.

Checks
- `prepare_intel.py` re-run on the raw archive (sha256 21d15688…) gives a matrix identical to
  `inputs/intel_time.npy` (same matrix SHA-256).
- `figures_pdf.py` reproduces the four version-1.0 figures pixel for pixel.
- Figures 2, 6, 7 and 9 match the published figures point for point (jitter aside); the Section 6.9
  statistics match the text exactly.
- On the 33-mote layout the routing script reproduces the 1,849 rounds of earlier drafts for direct
  and shortest-path routing.
- `text_checks.py`: the anchor is the LEACH per-report energy (3.77e-4 J); no verdict changes between
  1.0e-4 and 7.5e-4 J (first change at 7.6e-4 J, taxi CNN-LSTM); open-loop test-span fraction 0.988 at 0.10 C;
  3 complete rows without interpolation.
- `keras_equivalence.py`: all 196 saved models agree with Keras within the 2e-4 tolerance; maximum relative
  error 6.5e-7, median 1.5e-7.
- `receiver_checks.py`: 264 runs (11 streams x 4 tolerances x 3 cheap arms x 2 spans), all passed; the
  suppression of every run is identical to `ms_numbers.json` on both spans.
- `epoch_counter_check.py`: every one of the 30 motes has 1 to 5 backward jumps of more than 1,000 counts in its
  valid readings, 83% of them in three events shared by 12 to 21 motes; readings sharing an epoch number lie
  up to 17.6 days apart (30 motes) and 34.0 days apart (33 motes); the withdrawn matrix is rebuilt with 99.98%
  cell agreement, and 21.7% of its rows combine readings more than 7 days apart (maximum 17.6 days).
- `interpolation_check.py`: the rebuilt matrix is identical to `intel_time.npy`; 227,237 of its 753,810 cells are
  interpolated, 3,896 of them from gaps longer than eight bins (differing from the last reading before the gap by
  0.018 C at the median, 1.92 C at most); 9 rows at the training-validation boundary and 8 at the
  validation-test boundary hold a value that depends on a reading across it, all within 155 s of the boundary.

Run order for the added files (continuing the numbering of `RUN_ORDER.md`)

8. `python3 figures_pdf.py`               -> fig1..fig4 .png and .pdf   (runs figures.py unmodified)
9. `python3 figures_extra.py`             -> fig_intel_matrix, fig_per_seed, fig_energy_axis, fig_checkpoints (.png, .pdf)
                                          and checkpoint_stats.json (Section 6.9); needs intel_time_rows.csv from step 1
10. `python3 routing_baselines.py`        -> routing_30motes.json (Table 5; seeds 0-19)
    `python3 routing_baselines.py --motes old33_motes.json --seeds 0,1,2,3,4`  -> routing_33motes.json (check against earlier drafts)
11. `python3 text_checks.py --raw raw/data.txt` -> text_checks.json (Section 5.2 anchor and verdict band, open-loop test-span fraction, stress test)
12. `python3 keras_equivalence.py`        -> keras_equivalence.json (NumPy vs Keras on all 196 saved models; TensorFlow 2.21)
13. `python3 receiver_checks.py`         -> receiver_checks.json (independent-receiver check of the cheap arms; needs the NAB files)
14. `python3 epoch_counter_check.py`     -> epoch_counter.json (Section 5.2; needs raw/data.txt)
15. `python3 interpolation_check.py`     -> interpolation_check.json (Section 5.2; needs raw/data.txt)
