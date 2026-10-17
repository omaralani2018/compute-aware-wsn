done_ds=" taxi traffic cpu "
while true; do
  for ds in machine_temp grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy; do
    case "$done_ds" in *" $ds "*) continue;; esac
    n=$(ls weights/${ds}_lightweight_s*.npz 2>/dev/null | wc -l)
    if [ "$n" = 20 ] && ! ps aux | grep -v grep | grep -q "retrain_lw.py $ds "; then
      python3 -I oracle_switch.py $ds lightweight > oracle_$ds.log 2>&1; done_ds="$done_ds$ds "; echo "oracle done $ds"
    fi
  done
  [ $(echo $done_ds | wc -w) = 10 ] && { echo ALLDONE; exit 0; }
  sleep 20
done
