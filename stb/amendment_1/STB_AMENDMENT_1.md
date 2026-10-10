# Amendment 1 to STB_PROTOCOL.md (version 4)

Dated 9 October 2026. Written after the run of section 7 had been completed and its results seen. The
run started after the freezing message sent to the co-author by WhatsApp at 02:43, 9 October 2026
(Asia/Baghdad); RUN_LOG.txt records it from the container restart at 02:53. The frozen protocol
(SHA-256 1610ae2ae633cb0f8429fdcd72e85a4578107ddd2301143c9afcd95970deec25) and the frozen code are
unchanged and remain the record. This amendment follows the preamble of the protocol: it states the
reason for each change and the results affected. It is kept as a separate file, rather than appended to
STB_PROTOCOL.md, so that the frozen hash still verifies.

## A1.1 Parsing order in stb_prepare.py (found by the co-author, M. M. Jaber)

Finding. In parse(), run_max is updated before a reading with an implausible temperature is rejected.
A rejected row with a timestamp far ahead of the others can therefore make later valid readings look
like a clock reset, so that they are dropped. This is a bug in the sense of the preamble: the protocol
(section 3, rule 1) defines the clock-reset rule on readings that are kept.

Correction. run_max is updated only after the temperature check:

    if not (T_MIN < b < T_MAX): out_range += 1; continue
    run_max = max(run_max, a)

No other line of stb_prepare.py changes.

Regression probe (probe_timestamp_order.py, synthetic data only): one station file of 1,020 valid
readings with one rejected row (999 degC) whose timestamp lies ahead of the last 20 readings.
Frozen code: 1,000 of 1,020 valid readings kept, 20 dropped as a clock reset. Corrected code: 1,020 of
1,020 kept. This reproduces the co-author's case.

Effect on the results: none.
- In the frozen run no reading of any of the 23 station files was dropped as a clock reset
  (stb_meta.json: backward_jumps = 0 and not_time_ordered = false for every station), so the order
  of the two checks could not change any kept reading.
- stb_prepare.py with the correction, run on the same stbernard-meteo.zip, gives byte-identical outputs
  (identity_and_probe.log):
  stb_meta.json        a5f7f1748307a73a0e71af1b572404291ae527134d2157573c02a6b9b138a777
  stb_segments_obs.pkl e3422d075f835f45071efab8ad69124dda8024e8e837f9bec9b0eef0555e40c0
  stb_segments_fill.pkl 5f8f468f13a5bd982ab1b83e116b63790a8f776e7423bc3252cc6c934ffeae76
- Every later step reads only these files, so no bug-fix rerun is needed and no output is invalidated.

stb_synthetic_check.py, run unchanged with the corrected stb_prepare.py and the other frozen files:
all checks pass (synthetic_check_amended.log).

## A1.2 Assumption of the primary test (raised by the co-author)

The exact sign-flip permutation test of section 5 enumerates all 2^20 sign assignments. It is an exact
test of the null hypothesis that the 20 per-seed differences are independent and each symmetric about
zero, so that their joint distribution does not change when signs are flipped. Independence follows
from the design, conditional on the fixed data: each seed is a separate training run, and all are
compared with the same deterministic low-cost reference. Symmetry about zero is an assumption, not a
consequence of the design, and the manuscript states it where the p-values are reported (Sections 5.5
and 6.10 of manuscript v4_105). The same applies to the secondary Wilcoxon signed-rank test.

The primary test is not replaced. Noted after the results were known, as a description and not as a new
method: in both preparations, for both learned models and for both quantities (dd and g), all 20 per-seed
differences are negative. A two-sided sign test, which assumes only independence and a zero median,
then gives p = 2 x 0.5^20 = 1.9 x 10^-6, the same value as the permutation test. The direction of every
comparison is therefore also supported by a test that needs no symmetry assumption, although that test
concerns the median of the per-seed differences rather than their mean, which is the estimand of
section 5.

## A1.3 Section numbers

The frozen files refer to sections of manuscript v4_104. In v4_105 the routing comparison (old Section
6.5 and Table 5) has left the manuscript, so old Sections 6.6 to 6.9 are now 6.5 to 6.8 and old
Tables 6 to 10 are now 5 to 9; Sections 6.9 and 6.10 and Tables 10 and 11 are new in v4_105. For example, "GAP-8 values of Section 6.6" in section 5 of the protocol,
and "Section 6.6, Table 6" in gap8_lstm_energy.py and platform_energy.py, now refer to Section 6.5 and
Table 5. The frozen files are not edited.

## A1.4 Files (SHA-256)

    9c3657579cdddf54fdbf04ae1fec3128a7d3f502fcda3ffbe98117a8898aa0af  stb_prepare.py (corrected)
    f6d0aab99e3ff0163c2329926f3c8ef3aa25729c57102045f316d3c614ebec57  probe_timestamp_order.py
    901f738ee3456f5537825002fb324f8a92cdf6b2d7417283e5415a32d1e4acc8  stb_prepare.py (frozen, unchanged)
    1610ae2ae633cb0f8429fdcd72e85a4578107ddd2301143c9afcd95970deec25  STB_PROTOCOL.md (unchanged)

The hashes of identity_and_probe.log, synthetic_check_amended.log and this file are listed in
AMENDMENT_1.sha256.
