#!/bin/bash
cd "${WORKDIR:-.}"   # the directory the runs were made in (see README)
python3 run_learned.py intel_time lightweight 0 20 > log_intel_time_lw.txt 2>&1
python3 run_learned.py intel_time cnn_lstm 0 20 > log_intel_time_cnn.txt 2>&1
python3 run_tau.py intel_time > tlog_intel_time.txt 2>&1
echo done >> log_intel_time_done.txt
