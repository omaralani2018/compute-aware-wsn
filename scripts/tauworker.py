import os, sys, json, subprocess
Q=["intel","machine_temp","twitter_aapl","taxi","grok_asg_anomaly","cpu","rds_cpu_util","occupancy","iio_networkin","exchange_cpc","traffic"]
for ds in Q:
    f=f"tau_{ds}.json"
    if os.path.exists(f) and len(json.load(open(f)))>=20: continue
    try: fd=os.open(f"tlock_{ds}",os.O_CREAT|os.O_EXCL)
    except FileExistsError: continue
    subprocess.run([sys.executable,"run_tau.py",ds],stdout=open(f"tlog_{ds}.txt","a"),stderr=subprocess.DEVNULL)
    os.close(fd); os.remove(f"tlock_{ds}")
