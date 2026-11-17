#!/bin/bash
# Stop only the current test-time-shift process so the queue reloads the new batching code.
# Does not touch training or the upload.
pid=$(ps -eo pid,cmd | grep 'final/infer_tta.py' | grep -v grep | awk '{print $1; exit}')
if [ -n "$pid" ]; then
  echo "kill $pid"
  kill "$pid"
else
  echo "no infer_tta"
fi
sleep 2
echo '--- infer ---'
ps -eo pid,cmd | grep -E 'infer_tta.py|queue_v3_infer.sh' | grep -v grep | sed 's/ --data.*//' || true
echo '--- train ---'
ps -eo cmd | grep -c 'extra_tag vnx010_twin_s2' || true
