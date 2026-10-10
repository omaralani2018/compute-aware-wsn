#!/bin/bash
cd "${WORKDIR:-.}"   # the directory the runs were made in (see README)
for i in $(seq 1 40); do
  rm -f lock_*
  python3 worker.py cnn_lstm & python3 worker.py cnn_lstm & wait
  left=$(python3 -c "
import json,glob
print(sum(max(0,20-sum(r['predictor']=='cnn_lstm' for r in json.load(open(f))['entries'])) for f in glob.glob('learned_*.json')))")
  echo "$(date +%T) round $i left=$left" >> bg_progress.txt
  [ "$left" = "0" ] && break
done
