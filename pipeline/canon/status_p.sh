#!/bin/bash
# Status: Lane P start-up, GPU, memory, rsync state.
sleep "${1:-0}"
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon"
tail -n 4 laneP.log | cut -c1-200
for t in vnx010_pv_ft vnx010_pv_ft2 second_twin_pv_s1; do
  echo "$t: $(grep -E 'Train:' ~/projects/detector-program/$t.log 2>/dev/null | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110)"
  grep -E 'Traceback|RuntimeError|Killed|out of memory' ~/projects/detector-program/$t.log 2>/dev/null | tail -n 2
done
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
free -g | sed -n 2,3p
echo "rsync: $(ps -o stat= -C rsync | sort | uniq -c | tr '\n' ' ')"
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
tail -n 3 ../final/p3_eval_pv.log
