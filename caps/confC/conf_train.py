import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.abspath(os.path.join(HERE, "../stage17/run"))
sys.path.insert(0, RUN); sys.path.insert(0, HERE)
import run_learned as rl, conf_data as cd
os.chdir(HERE)
ds, kind = sys.argv[1], sys.argv[2]
A, unit = cd.load(ds)
rl.run(A, ds, unit, [kind], range(20), f"learned_{ds}.json")
