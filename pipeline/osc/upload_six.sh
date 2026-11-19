#!/bin/bash
# 6 parallel rsync shards at a time (the 16/32-shard runs got connection resets). Twin first, then canon.
# rsync --size-only skips files already on OSC, so this only sends what is missing.
cd "$(dirname "$0")"
N=6
D="${BDC_DATA}"
STAGE="${BDC_OUT}/upload_stage"
mkdir -p "$STAGE"
rm -f upload_shard_*.log
for ds in claude_twin_v1 claude_canon_all_v1; do
  rm -f "$STAGE/$ds.shard."*
  ( cd "$D/$ds" && find . \( -type f -o -type l \) ! -path '*/__pycache__/*' | sed 's|^\./||' | awk -v n="$N" -v p="$STAGE/$ds.shard" '{fn=sprintf("%s.%d",p,NR%n); print >> fn; close(fn)}' )
  pids=()
  for i in $(seq 0 $((N-1))); do
    bash ./upload_shard.sh "$ds" "$STAGE/$ds.shard.$i" "$i" </dev/null >/dev/null 2>&1 &
    pids+=($!)
    sleep 2
  done
  wait "${pids[@]}"
  echo "DATASET_DONE $ds $(date -Is)" >> upload_six.log
done
echo "UPLOAD_ALL_DONE $(date -Is)" >> upload_six.log
