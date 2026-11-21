# NLMS dual predictor (Table 5, Section 5.6)

Added in version 1.8. `RULE6.txt` was written before any NLMS number was computed. `nlms_zoh.py` runs the NLMS
dual predictor of [6] (initialised at the zero-order hold) under the protocol of `src/matched_baselines.py`, with
the published setting and with p and µ selected on validation, and checks every run with the independent receiver.
`RESULTS6.sha256` holds the hashes of the script and of `nlms_results.json` as first produced.

Run from the `run/` folder of the top-level `README.md`:

```bash
cd run && python3 ../caps/nlms/nlms_zoh.py      # -> caps/nlms/nlms_results.json (about 10 s)
```
The output equals the shipped file byte for byte.
