#!/bin/bash
cd "$(dirname "$0")"
if pgrep -f 'queue_v3_infer.sh' >/dev/null; then
  echo already
  pgrep -af 'queue_v3_infer.sh'
  exit 0
fi
setsid nohup bash ./queue_v3_infer.sh </dev/null >/dev/null 2>&1 &
echo "pid $!"
