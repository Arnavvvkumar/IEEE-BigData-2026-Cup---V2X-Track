#!/bin/bash
# Stop only the v3 inference queue. Does not touch vnx010_twin_s2 training or the OSC upload.
set -u
ps -eo pid,cmd | grep -E 'queue_v3_infer.sh|run_infer_dir.sh|infer_test.py' | grep -v grep | awk '{print $1}' > /tmp/v3_infer_pids
if [ -s /tmp/v3_infer_pids ]; then
  kill $(cat /tmp/v3_infer_pids) || true
  sleep 2
  kill -9 $(cat /tmp/v3_infer_pids) 2>/dev/null || true
fi
echo '--- still inference ---'
ps -eo pid,cmd | grep -E 'queue_v3_infer.sh|run_infer_dir.sh|infer_test.py' | grep -v grep || echo none
echo '--- training ---'
ps -eo pid,cmd | grep 'extra_tag vnx010_twin_s2' | grep -v grep | wc -l
