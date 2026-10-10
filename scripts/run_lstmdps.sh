#!/bin/bash
cd "$(dirname "$0")"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1
python3 lstmdps_protocol.py 0,1,2,3,4 $(seq -s, 0 14) lstmdps_intel_A.jsonl > lstm_A.log 2>&1 &
python3 lstmdps_protocol.py 0,1,2,3,4 $(seq -s, 15 29) lstmdps_intel_B.jsonl > lstm_B.log 2>&1 &
wait; echo DONE > lstm_done.flag
