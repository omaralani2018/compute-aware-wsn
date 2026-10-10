#!/bin/bash
# 9 Oct 2026 ~22:02 Asia/Baghdad: the second computer was not used; its four streams run here, on disjoint lists.
cd "$(dirname "$0")"
bash tm_driver.sh machine_temp iio_networkin > driverC.out 2>&1 &
bash tm_driver.sh twitter_aapl cpu > driverD.out 2>&1; wait
