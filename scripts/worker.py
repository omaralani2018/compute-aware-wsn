"""Pull datasets from a shared queue; one worker per dataset file at a time."""
import os, sys, json, subprocess
QUEUE = ["intel","machine_temp","twitter_aapl","taxi","grok_asg_anomaly","cpu",
         "rds_cpu_util","occupancy","iio_networkin","exchange_cpc"]
KIND = sys.argv[1] if len(sys.argv) > 1 else "cnn_lstm"
def done(ds):
    f=f"learned_{ds}.json"
    if not os.path.exists(f): return False
    return sum(r["predictor"]==KIND for r in json.load(open(f))["entries"]) >= 20
for ds in QUEUE:
    if done(ds): continue
    try: fd=os.open(f"lock_{ds}", os.O_CREAT|os.O_EXCL)
    except FileExistsError: continue
    subprocess.run([sys.executable,"run_learned.py",ds,KIND,"0","20"],
                   stdout=open(f"log_{ds}.txt","a"), stderr=subprocess.DEVNULL)
    os.close(fd); os.remove(f"lock_{ds}")
