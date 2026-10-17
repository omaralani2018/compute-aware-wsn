import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"appliances_indoor": None, "occupancy_office": None}
def load(name, split0=0.50):
    if name == "appliances_indoor":
        d = pd.read_csv(os.path.join(HERE, "data", "energydata_complete.csv"))
        d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date")
        A = d[["T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9"]].to_numpy(float)
    else:
        parts = [pd.read_csv(os.path.join(HERE, "data", f), sep=",") for f in ("datatest.txt", "datatraining.txt", "datatest2.txt")]
        d = pd.concat(parts); d["date"] = pd.to_datetime(d["date"])
        A = d.groupby("date")["Temperature"].mean().sort_index().to_numpy(float)[:, None]
    if not np.isfinite(A).all(): raise ValueError(name + " has non-finite values")
    return A, 1.0                                         # physical tolerance in degC, as Intel
