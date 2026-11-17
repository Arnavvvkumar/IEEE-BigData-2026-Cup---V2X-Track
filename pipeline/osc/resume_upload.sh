#!/bin/bash
# Resume the per-file rsync. It already fills this network; do not start a second copy.
cd "$(dirname "$0")"
if pgrep -f 'osc/upload_rsync.sh' >/dev/null || pgrep -f 'rsync -aL --partial --inplace -z' >/dev/null; then
  echo "uploader already running"
  pgrep -af 'upload_rsync.sh|rsync -aL --partial'
  exit 0
fi
setsid nohup bash ./upload_rsync.sh >> ./upload_rsync.log 2>&1 </dev/null &
echo "pid $!"
sleep 1
pgrep -af 'upload_rsync.sh|rsync -aL --partial' || echo "failed to start"
