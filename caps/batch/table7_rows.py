"""Rows of Table 7 (batched reporting at D = 25) from dev_batch.json and heldout_batch.json.
Cheaper: S1 or S2, whichever has the lower radio energy per reading (checked on both node types);
Saving: 1 - E(cheaper)/E(S0) per node type. Writes table7_rows.json."""
import os, json
H = os.path.dirname(os.path.abspath(__file__))
NAME = {"intel_time": "Intel temperature", "appliances_indoor": "Indoor [38]", "occupancy_office": "Indoor [39]"}
rows = []
for grp, f in (("dev", "dev_batch.json"), ("further", "heldout_batch.json")):
    for ds, r in json.load(open(os.path.join(H, f))).items():
        ch = {}
        for nd in ("M3", "nRF52840"):
            s1, s2 = r[nd]["S1"]["25"], r[nd]["S2"]["25"]
            ch[nd] = ("S1", s1) if s1 < s2 else ("S2", s2)
        assert ch["M3"][0] == ch["nRF52840"][0], ds
        sav = ["%d%%" % round(100 * (1 - ch[nd][1] / r[nd]["S0"])) for nd in ("M3", "nRF52840")]
        rows.append([grp, [NAME.get(ds, ds), "%.3f" % r["rho"], ch["M3"][0]] + sav])
json.dump(rows, open(os.path.join(H, "table7_rows.json"), "w"))
for g, r in rows: print(g, *r)
