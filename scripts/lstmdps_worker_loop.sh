#!/bin/bash
# re-runs the worker until it exits cleanly; the worker resumes from its jsonl
cd "$(dirname "$0")"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1
until python3 lstmdps_protocol.py "$1" "$2" "$3" >> "$4" 2>&1; do sleep 3; done
