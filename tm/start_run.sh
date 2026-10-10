#!/bin/bash
# TM_PROTOCOL.md section 7, run after the freeze message of 9 Oct 2026 (~18:44 Asia/Baghdad).
cd "$(dirname "$0")"
sha256sum -c TM_FROZEN.sha256 > run_hashcheck.log 2>&1 || { echo "HASH CHECK FAILED" >> run_hashcheck.log; exit 1; }
PYTHONDONTWRITEBYTECODE=1 python3 tm_synthetic_check.py > run_synthetic_check.log 2>&1 || { echo "SYNTHETIC CHECK FAILED" >> run_synthetic_check.log; exit 1; }
bash tm_driver.sh intel_time taxi traffic cpu machine_temp > driverA.out 2>&1 &
bash tm_driver.sh grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy > driverB.out 2>&1; wait
python3 tm_stats.py > run_stats.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> run_stats.log
