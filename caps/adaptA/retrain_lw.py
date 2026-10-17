import os,sys,json
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
sys.path.insert(0, os.path.abspath("../stage17/run"))
import run_learned as rl, nab_data as nd, numpy as np
ds, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
A, unit = (np.load("intel_time.npy"), 1.0) if ds=="intel_time" else nd.load(ds)
rl.run(A, ds, unit, ["lightweight"], range(s0,s1), f"lw_{ds}.json")
