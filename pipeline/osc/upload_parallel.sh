#!/bin/bash
# Replace the single rsync with several at once. The speed test reached 38 Mbps
# with multiple connections; one stream to the login node stays near 2 Mbps.
set -euo pipefail
cd "$(dirname "$0")"
N=16
D="$HOME/projects/bdc2026-track-b/data"
STAGE="$HOME/projects/bdc2026-track-b/upload_stage"
root="$(pwd)"
mkdir -p "$STAGE"
echo "stopping single uploader $(date -Is)"
pkill -f 'bash ./upload_rsync.sh' || true
pkill -f 'bash upload_rsync.sh' || true
sleep 1
pkill -f 'rsync -aL --partial --inplace -z' || true
sleep 2
if pgrep -f 'rsync -aL --partial --inplace -z' >/dev/null; then
  pkill -9 -f 'rsync -aL --partial --inplace -z' || true
fi
echo "single uploader stopped"
echo "stopping current shards $(date -Is)"
pkill -f 'bash ./upload_shard.sh' || true
sleep 1
pkill -f 'rsync -aL --partial --size-only' || true
sleep 2
if pgrep -f 'rsync -aL --partial --size-only' >/dev/null; then
  pkill -9 -f 'rsync -aL --partial --size-only' || true
fi
echo "shards stopped"
for ds in claude_twin_v1 claude_canon_all_v1; do
  echo "listing $ds"
  rm -f "$STAGE/$ds.shard."*
  cd "$D/$ds"
  find . \( -type f -o -type l \) ! -path '*/__pycache__/*' | sed 's|^\./||' | awk -v n="$N" -v p="$STAGE/$ds.shard" '{
    fn = sprintf("%s.%d", p, NR % n)
    print >> fn
    close(fn)
  }'
  cd "$root"
  for i in $(seq 0 $((N - 1))); do
    list="$STAGE/$ds.shard.$i"
    if [ ! -s "$list" ]; then
      echo "empty $list"
      continue
    fi
    echo "start $ds shard $i files $(wc -l < "$list")"
    setsid nohup bash ./upload_shard.sh "$ds" "$list" "$i" </dev/null >/dev/null 2>&1 &
    sleep 1
  done
done
sleep 2
pgrep -af 'upload_shard.sh|rsync -aL --partial --size-only' || echo "no shards running"
echo "PARALLEL_STARTED $(date -Is)"
