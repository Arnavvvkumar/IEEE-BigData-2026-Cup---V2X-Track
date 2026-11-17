#!/bin/bash
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/osc"
if pgrep -f 'upload_canon_12.sh' >/dev/null; then
  echo already
  exit 0
fi
setsid nohup bash ./upload_canon_12.sh </dev/null >/dev/null 2>&1 &
echo "pid $!"
