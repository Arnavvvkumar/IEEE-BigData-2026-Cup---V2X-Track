#!/bin/bash
# What is alive after a possible lock/sleep: training, scoring, Globus, GPU, memory, clock.
date -Is
echo "uptime: $(uptime -p)"
echo "train.py procs: $(pgrep -fc 'train.py')  infer: $(pgrep -fc 'infer_test.py')  eval150: $(pgrep -fc 'eval150.py')"
ps -eo pid,stat,etime,cmd | grep -E 'train.py|infer_test|eval150|globusconnect|p6_eval|laneQ' | grep -v grep | cut -c1-110 | head -n 8
echo "rsync states: $(ps -o stat= -C rsync | sort | uniq -c | tr '\n' ' ')"
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
grep -E 'Train:' ${BDC_PROGRAM}/vnx010_pv_ft3.log | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-110
tail -n 2 "${BDC_ROOT}/experiments/claude_carla_overnight/final/p6_eval_ft2.log" | cut -c1-220
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -g | sed -n 2p
