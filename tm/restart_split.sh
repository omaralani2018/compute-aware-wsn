#!/bin/bash
# 9 Oct 2026 ~21:40 Asia/Baghdad: the remaining streams are split between this machine and a second computer
# (same software versions, same frozen files), on disjoint stream lists (protocol section 5).
# This machine: taxi traffic | rds_cpu_util exchange_cpc occupancy.
# Second computer: cpu machine_temp iio_networkin twitter_aapl.
cd "$(dirname "$0")"
sha256sum -c TM_FROZEN.sha256 > restart_hashcheck.log 2>&1 || { echo "HASH CHECK FAILED" >> restart_hashcheck.log; exit 1; }
bash tm_driver.sh taxi traffic > driverA3.out 2>&1 &
bash tm_driver.sh rds_cpu_util exchange_cpc occupancy > driverB3.out 2>&1; wait
