# Training on actual versus reconstructed history: results

Generated from `tm_summary.json` (written by `tm_stats.py` under the frozen protocol `TM_PROTOCOL.md`, version 3,
frozen on 9 October 2026; see `stb/FREEZE_RECORD.md`). Lightweight model, eleven streams, seeds 0 to 9,
tau = 0.10. d is test suppression of arm C (retrained on reconstructed history) minus arm B+ (retrained on
actual history, same budget), mean over seeds, with its 95% percentile bootstrap interval over seeds
(descriptive). p is the exact two-sided sign-flip p-value of the mean (sign-exchangeability assumed); the
threshold is 0.05/11 = 0.0045. Net saving g is per node-round against the validation-selected low-cost
reference of the main study, at e_r = 3.8×10⁻⁴ J and 1 nJ per MAC, mean over the same 10 seeds, with its label
from the same test ("phase1" is the main-study checkpoint of each seed, as context).

| Stream | d | 95% interval | p | Outcome | g phase1 (µJ) | g B+ (µJ) | g C (µJ) |
|---|---|---|---|---|---|---|---|
| intel_time | −0.0756 | [−0.1391, −0.0159] | 0.0566 | inconclusive | −11.7 (costs) | −28.1 (ns) | −56.8 (costs) |
| taxi | +0.0049 | [−0.0018, +0.0128] | 0.2773 | inconclusive | +26.7 (saves) | +26.8 (saves) | +28.7 (saves) |
| traffic | −0.0027 | [−0.0091, +0.0032] | 0.5195 | inconclusive | +6.9 (saves) | +2.7 (ns) | +1.7 (ns) |
| cpu | −0.0117 | [−0.0237, +0.0002] | 0.1094 | inconclusive | −35.2 (costs) | −55.2 (costs) | −59.6 (costs) |
| machine_temp | +0.0017 | [−0.0031, +0.0065] | 0.5137 | inconclusive | −15.8 (costs) | −10.6 (costs) | −9.9 (ns) |
| grok_asg_anomaly | +0.0014 | [−0.0009, +0.0037] | 0.3320 | inconclusive | −94.5 (costs) | −97.2 (costs) | −96.7 (costs) |
| rds_cpu_util | +0.0006 | [−0.0013, +0.0021] | 0.6250 | inconclusive | −47.8 (costs) | −48.8 (costs) | −48.6 (costs) |
| iio_networkin | +0.0209 | [+0.0021, +0.0378] | 0.0547 | inconclusive | −75.5 (costs) | −48.0 (costs) | −40.0 (costs) |
| twitter_aapl | +0.0246 | [+0.0124, +0.0369] | 0.0059 | inconclusive | −32.4 (costs) | −35.2 (costs) | −25.8 (costs) |
| exchange_cpc | −0.0033 | [−0.0084, +0.0023] | 0.3125 | inconclusive | +15.9 (saves) | +17.0 (saves) | +15.8 (saves) |
| occupancy | +0.0008 | [−0.0046, +0.0063] | 0.7949 | inconclusive | −13.6 (costs) | −9.5 (costs) | −9.2 (costs) |

All 110 runs are valid: every receiver check passed in both arms and in phase 1, and phase 1 reproduced the
main-study entry of its seed exactly (selected checkpoint, validation and test suppression) in all 110 runs. No seed is missing. The outcome is inconclusive on
every stream; this is not evidence that training on reconstructed history has no effect, and the result is
limited to one round of reconstructed-history retraining of the lightweight model. No energy label changes
from "saves" to "costs" or back between B+ and C; on two streams one arm's label is "ns" where the other's is
"costs" (intel_time: B+ ns, C costs; machine_temp: B+ costs, C ns).

## Run record

- 9 October 2026, after the freeze message of about 18:44 (Asia/Baghdad): `start_run.sh` checked the frozen
  hashes, ran the synthetic checks (passed) and started the two drivers of protocol section 7.
- The cloud machine running the experiment was suspended twice (at about 19:33 and 19:43). Each time the
  drivers were restarted with the same commands after the frozen hashes were checked again
  (`restart_run.sh`); the attempts cut off are logged as interrupted and not counted (protocol section 5),
  and completed seeds were skipped.
- At about 21:40 the drivers were stopped by hand and restarted on disjoint stream lists
  (`restart_split.sh`, then `run_rest.sh`), to run more streams at once; the two seeds running at that
  moment are logged as interrupted and not counted. A second computer was prepared but not used: every run
  was made on this machine with the software versions of protocol section 7.
- `tm_stats.py` was run after the last run, unchanged (its hash is in `TM_FROZEN.sha256`).

`tm_runs/log.txt` is the driver log (the absolute path of the run directory is replaced by
`<run directory>`); `tm_runs/` holds the result file and the attempt count of every (stream, seed).
