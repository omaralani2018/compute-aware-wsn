#!/bin/bash
cd "$(dirname "$0")"
run(){ for d in "$@"; do python3 matched_baselines.py $d >> log_$d.txt 2>&1; done; }
run taxi traffic cpu machine_temp grok_asg_anomaly &
run intel_time rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy &
wait; echo ALLDONE > done.flag
