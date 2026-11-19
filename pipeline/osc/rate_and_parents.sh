#!/bin/bash
echo "=== parents ==="
ps -eo pid,ppid,etime,cmd | grep -E 'upload_six|upload_shard.sh claude_canon' | grep -v grep
echo "=== rate ==="
bash "${BDC_ROOT}/experiments/claude_carla_overnight/osc/parallel_rate.sh"
