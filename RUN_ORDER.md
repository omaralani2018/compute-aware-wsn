# Latest code as run (26 Sep 2026): run order

All results in the rebuilt manuscript were produced by these files, in this order,
from the working directory with the NAB repository cloned at `nab_repo/` (commit ea702d75)
and the raw Intel file at `intel_raw/raw/data.txt` (sha256 21d15688…).

1. `python3 prepare_intel.py`            -> intel_time.npy, intel_time_meta.json   (Intel, timestamp-indexed)
2. `python3 -c "import frozen_benchmark as fb, numpy as np; fb.run_dataset(np.load('intel_time.npy'),'intel_time',1.0,False,'bench_intel_time.json')"`
                                          (cheap arms; NAB baselines likewise via nab_data.load)
3. `python3 run_learned.py <ds> lightweight 0 20`  and  `python3 run_learned.py <ds> cnn_lstm 0 20`
                                          -> learned_<ds>.json   (ds = intel_time or a NAB name)
   `worker.py` / `bg.sh` / `bg_intel.sh` are the schedulers that ran these in the background,
   two workers at a time; they add nothing to the protocol.
4. `python3 run_tau.py <ds>`              -> tau_<ds>.json   (lightweight tolerance sweep; `tauworker.py` schedules)
5. `python3 ms_numbers.py`                -> ms_numbers.json  (every number in the manuscript)
6. `python3 figures.py`                   -> fig1..fig4 .png
7. `python3 build_ms.py`                  -> rebuilt.docx from base.docx + ms_numbers.json + figures

Protocol constants live only in frozen_benchmark.py (split, tau, look-back, epoch cap, checkpoint
interval, seeds, window cap, grids, tie-breaks, energy constants, MAC counts, INTEL_SPEC="time").
The evaluator is dual_prediction_fix.py (sha256 e8cfb5f6…, unchanged since 21 Sep).
fastfwd.py is the NumPy forward pass asserted equal to Keras at every checkpoint.
Environment: Python 3.11, NumPy, SciPy, pandas, TensorFlow 2.21, python-docx, matplotlib; 2 CPU threads.
