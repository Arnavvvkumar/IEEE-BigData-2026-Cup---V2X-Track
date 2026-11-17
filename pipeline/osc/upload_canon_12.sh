#!/bin/bash
# Canon only, 12 rsync streams. Twin is already on OSC and is not touched.
# 12 was the fastest stable count. 16 and 32 started resetting the SSH sessions.
set -uo pipefail
cd "$(dirname "$0")"
N=12
ds=claude_canon_all_v1
D="$HOME/projects/bdc2026-track-b/data"
STAGE="$HOME/projects/bdc2026-track-b/upload_stage"
log=upload_canon_12.log
echo "STOP_SIX $(date -Is)" >> "$log"

# Stop the parent first so it cannot record the canon set as finished.
pkill -f 'bash ./upload_six.sh' || true
sleep 1
pkill -f 'upload_shard.sh claude_canon_all_v1' || true
pkill -f 'claude_canon_all_v1.shard' || true
sleep 2
pkill -9 -f 'upload_shard.sh claude_canon_all_v1' || true
pkill -9 -f 'claude_canon_all_v1.shard' || true
sleep 1
if pgrep -f 'claude_canon_all_v1.shard' >/dev/null; then
  echo "OLD_CANON_STILL_RUNNING $(date -Is)" >> "$log"
  exit 1
fi

mkdir -p "$STAGE" old_canon_logs
mv upload_shard_claude_canon_all_v1_*.log old_canon_logs/ 2>/dev/null || true
rm -f "$STAGE/$ds.shard."*
( cd "$D/$ds" && find . \( -type f -o -type l \) ! -path '*/__pycache__/*' | sed 's|^\./||' | awk -v n="$N" -v p="$STAGE/$ds.shard" '{fn=sprintf("%s.%d",p,NR%n); print >> fn; close(fn)}' )
echo "SPLIT $(date -Is) lists $(ls "$STAGE/$ds.shard."* | wc -l)" >> "$log"

for i in $(seq 0 $((N - 1))); do
  setsid nohup bash ./upload_shard.sh "$ds" "$STAGE/$ds.shard.$i" "$i" </dev/null >/dev/null 2>&1 &
  sleep 2
done
echo "STARTED_12 $(date -Is)" >> "$log"
echo "rsyncs $(pgrep -fc 'rsync -aL' || true)" >> "$log"
