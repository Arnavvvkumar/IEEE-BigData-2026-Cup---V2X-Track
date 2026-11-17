#!/bin/bash
echo "=== parents ==="
ps -eo pid,ppid,etime,cmd | grep -E 'upload_six|upload_shard.sh claude_canon' | grep -v grep
echo "=== rate ==="
bash "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc/parallel_rate.sh"
