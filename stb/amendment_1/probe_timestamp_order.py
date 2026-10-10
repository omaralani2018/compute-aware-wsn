"""Regression probe for amendment 1 (co-author's finding of 9 Oct 2026): a row with an out-of-range
temperature and a far-future timestamp must not advance run_max, so later valid readings are not
dropped as clock resets. Builds a synthetic station file; usage: python3 probe_timestamp_order.py <dir with stb_prepare.py>"""
import sys, importlib.util
spec = importlib.util.spec_from_file_location("prep", sys.argv[1] + "/stb_prepare.py"); prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
t0 = 1.19e9                                                     # September 2007, inside the protocol window
def row(t, temp): return " ".join(["7"] + ["0"] * 6 + [f"{t:.0f}", f"{temp}"] + ["0"] * 8)   # 17 fields, station 7
rows = [row(t0 + 120 * i, 5.0) for i in range(1000)]
rows.append(row(t0 + 120 * 1020 + 3600, 999.0))                 # rejected temperature, timestamp ahead of the next 20 readings
rows += [row(t0 + 120 * (1000 + i), 5.0) for i in range(20)]   # 20 drops stay under the 5% fallback, so the rule is applied
r = prep.parse("\n".join(rows).encode(), 7)
kept = len(r["t"]); print("valid readings kept:", kept, "of 1020;", "backward drops:", r["backward_jumps"])
sys.exit(0 if kept == 1020 else 1)
