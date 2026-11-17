#!/bin/bash
# One-shot status of the overnight jobs.
free -g | sed -n 2,3p
for t in second_twin_st_s1 vnx010_pv_ft second_twin_pv_s1; do
  echo "$t: $(grep -E 'Train:' ~/projects/detector-program/$t.log 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110)"
  grep -E 'Traceback|RuntimeError|Killed|out of memory' ~/projects/detector-program/$t.log 2>/dev/null | tail -n 2
done
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
tail -n 2 laneM.log | cut -c1-200
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
dmesg 2>/dev/null | grep -i -E 'out of memory|oom-kill' | tail -n 2
