#!/bin/bash
# Read-only view of Codex's overnight SECOND runner: process, start time, stdout target, recent output.
for p in $(pgrep -f run_second_overnight.sh); do
  echo "runner pid $p started $(ps -o lstart= -p $p)"
  ls -l /proc/$p/fd/1 2>/dev/null | awk '{print "stdout ->", $NF}'
  out=$(readlink /proc/$p/fd/1)
  [ -f "$out" ] && tail -5 "$out"
done
pgrep -af '[p]ython.*(train\.py|evaluate_openpcdet\.py)' | cut -c1-200
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
tail -c 600 ~/projects/detector-program/second_overnight_control.log | tr '\r' '\n' | grep -E 'Train|Skipping|rror' | tail -1 | cut -c1-200
