# Earlier runs of `matched_baselines.py`

These are the two runs that preceded the final one in `results/matched_*.json`. They are
kept so that the grid extensions can be checked. The code was the same in all three runs;
only the grids of the step-size constant `c` differed. `N ∈ {4, 5, 6, 8, 10}` and
`(w_f, w_s) ∈ {(4, 8), (5, 10)}` were the same throughout.

| Run | LMS `c` (µ = c / E_x) | AM-DR `c` (α = c / σ²) |
|---|---|---|
| `grid1/` | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1 (7 values) | 0, 1e-4, 1e-3, 1e-2, 0.1, 1 (6 values) |
| `grid2/` | 1e-4 … 1 (9 values) | 0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100 (8 values) |
| final (`results/`) | 1e-6 … 1 (13 values) | as `grid2/` |

Each run scored both validation and test. Every extension was triggered by a validation
selection at a grid edge, not by test results:

- `grid1/`: validation selected the upper LMS edge (c = 0.1) on four streams (intel_time,
  taxi, grok_asg_anomaly, exchange_cpc; single-error rule), the upper AM-DR edge (c = 1) on
  three (intel_time, machine_temp, iio_networkin), and the lower LMS edge (c = 1e-4) on
  twitter_aapl. The upper edges were extended, LMS to c = 1, the convergence bound of
  Eq. (5) in [13], and AM-DR by two decades.
- `grid2/`: the lower LMS edge (c = 1e-4) was still selected on twitter_aapl, so the LMS
  grid was extended down to 1e-6.
- Final grids: the remaining edge selections are ties at zero validation suppression,
  which go to the first grid entry, or α = 0 (AM-DR on grok_asg_anomaly).

Each folder holds the eleven `matched_*.json` and the console logs of that run.
