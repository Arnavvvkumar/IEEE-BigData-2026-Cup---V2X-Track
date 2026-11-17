#!/bin/bash
# Quick read-only state: what is actually running right now.
date
echo "training: $(pgrep -fa train.py | grep -o 'extra_tag [a-z0-9_]*' | sort -u | tr '\n' ' ')"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "upload_rsync: $(pgrep -fc 'upload_rsync.sh')  rsync procs: $(pgrep -fc 'rsync -aL')  push_code: $(pgrep -fc 'push_code.sh')"
tail -n 2 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_rsync.log" | cut -c1-140
tail -n 3 "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/push_code2.log" | cut -c1-140
tail -c 300 ~/projects/detector-program/vnx010_twin_s2.log | tr '\r' '\n' | tail -n 1 | cut -c1-150
echo "other python jobs: $(pgrep -fc 'infer_|eval150|analyze_dev|fuse')"
