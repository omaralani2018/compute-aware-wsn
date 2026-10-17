import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"ambient_temperature": "ambient_temperature_system_failure.csv", "occupancy_t4013": "occupancy_t4013.csv",
         "speed_6005": "speed_6005.csv", "speed_t4013": "speed_t4013.csv", "TravelTime_387": "TravelTime_387.csv",
         "TravelTime_451": "TravelTime_451.csv"}
def load(name, split0=0.50):
    v = pd.read_csv(os.path.join(HERE, "data", FILES[name]))["value"].to_numpy(float)
    A = v[:, None]
    if not np.isfinite(A).all(): raise ValueError(name + " has non-finite values")
    return A, float(A[:int(split0 * len(A))].std())          # same rule as nab_data.load
