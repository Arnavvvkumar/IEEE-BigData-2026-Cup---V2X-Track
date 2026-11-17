#!/bin/bash
echo "=== train procs ==="
ps -eo pid,etime,cmd | grep train.py | grep -v grep | head -n 5 || echo none
echo "=== log tail ==="
tail -n 8 "$HOME/projects/detector-program/vnx010_twin_s2.log" | tr '\r' '\n' | tail -n 8
echo "=== canon shard exits ==="
grep -hE 'rsync_exit|SHARD_' "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/upload_shard_claude_canon_all_v1_"*.log | tail -n 24
