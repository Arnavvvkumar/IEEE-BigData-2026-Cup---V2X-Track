#!/bin/bash
# Detach the single-stream upload so it survives this shell exiting.
cd "$(dirname "$0")"
if pgrep -f 'osc/upload_fast.sh' >/dev/null; then
  echo "already running"
  pgrep -af 'osc/upload_fast.sh'
  exit 0
fi
setsid nohup bash ./upload_fast.sh </dev/null >/dev/null 2>&1 &
echo "pid $!"
sleep 1
pgrep -af 'osc/upload_fast.sh' || echo "failed to start"
