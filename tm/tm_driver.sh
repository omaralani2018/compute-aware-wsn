#!/bin/bash
# TM_PROTOCOL.md, section 5: one process per (stream, seed); a seed gets at most two completed attempts.
# Before any attempt, the flock command, the software versions and every listed stream are checked (python3
# tm_run.py --check); a setup error stops the driver without counting attempts. An attempt is completed when
# tm_run.py returns (any exit status); it is then counted in .attempts and logged with its exit status. An
# attempt whose return the driver did not see (after an interruption of the machine or of the driver) leaves
# its .running marker; it is logged as interrupted on the next start of the driver and is not counted
# (runs are deterministic, so a repeat cannot select on outcomes). A seed with a result file is
# skipped, so the driver can simply be started again. Each seed is guarded by a kernel lock (flock) that
# the worker process inherits, so a seed whose worker is still running (even if its driver was killed) is
# skipped, and a lock is never left behind by an interruption. Use disjoint stream lists for parallel
# drivers. Run on storage that survives a restart of the machine.
# Usage: bash tm_driver.sh [stream ...]      (default: all eleven streams, seeds 0-9)
STREAMS="${*:-intel_time taxi traffic cpu machine_temp grok_asg_anomaly rds_cpu_util iio_networkin twitter_aapl exchange_cpc occupancy}"
mkdir -p tm_runs
log() { echo "$(date -u +%FT%TZ) $*" >> tm_runs/log.txt; }
if ! command -v flock >/dev/null || ! python3 tm_run.py --check $STREAMS >> tm_runs/log.txt 2>&1; then
  log "SETUP CHECK FAILED for: $STREAMS (no attempt started)"; exit 1
fi
for ds in $STREAMS; do
  for sd in 0 1 2 3 4 5 6 7 8 9; do
    base=tm_runs/tm_${ds}_s${sd}; out=$base.json; att=$base.attempts; run=$base.running
    [ -f "$out" ] && continue
    exec 9>"$base.flock"
    if ! flock -n 9; then log "LOCKED $ds s$sd (its worker is still running; skipped)"; exec 9>&-; continue; fi
    if [ -f "$out" ]; then exec 9>&-; continue; fi
    [ -f "$run" ] && { log "INTERRUPTED $ds s$sd (not counted)"; rm -f "$run"; }
    n=$(cat "$att" 2>/dev/null); [[ "$n" =~ ^[0-9]+$ ]] || n=0
    while [ ! -f "$out" ] && [ "$n" -lt 2 ]; do
      touch "$run"; log "start $ds s$sd attempt $((n + 1))"
      python3 tm_run.py "$ds" "$sd" >> tm_runs/log.txt 2>&1; rc=$?
      n=$((n + 1)); echo "$n" > "$att.tmp" && mv "$att.tmp" "$att"; rm -f "$run"
      log "EXIT $ds s$sd attempt $n rc=$rc"
    done
    [ -f "$out" ] || log "MISSING $ds s$sd after $n completed attempts"
    exec 9>&-
  done
done
