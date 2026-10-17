for ds in taxi traffic cpu machine_temp grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy; do python3 -I oracle_switch.py $ds cnn_lstm; done
