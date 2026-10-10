#!/bin/bash
# Restart after the machine was restarted at about 19:33 (Asia/Baghdad), 9 Oct 2026: same commands as
# start_run.sh (protocol section 7); completed seeds are skipped, interrupted attempts are logged and not counted.
cd "$(dirname "$0")"
sha256sum -c TM_FROZEN.sha256 > restart_hashcheck.log 2>&1 || { echo "HASH CHECK FAILED" >> restart_hashcheck.log; exit 1; }
bash tm_driver.sh intel_time taxi traffic cpu machine_temp > driverA2.out 2>&1 &
bash tm_driver.sh grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy > driverB2.out 2>&1; wait
python3 tm_stats.py > run_stats.log 2>&1
echo "DONE $(date -u +%FT%TZ)" >> run_stats.log
