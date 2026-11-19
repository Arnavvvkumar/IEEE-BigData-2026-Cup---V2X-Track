#!/bin/bash
# The Ascend login node is not reading the upload (round trip is about a minute
# and each stream is stuck near 100 kbps). OSC's transfer host is sftp.osc.edu
# and it sees the same scratch disk. Stop only the canon rsyncs, try that host,
# and if it does not answer, put the canon upload back on the login node.
set -uo pipefail
cd "$(dirname "$0")"
ds=claude_canon_all_v1
D="${BDC_DATA}"
STAGE="${BDC_OUT}/upload_stage"
log=upload_canon_12.log
echo "SWITCH $(date -Is)" >> "$log"

pkill -f 'upload_shard.sh claude_canon_all_v1' || true
pkill -f 'claude_canon_all_v1.shard' || true
sleep 2
pkill -9 -f 'upload_shard.sh claude_canon_all_v1' || true
pkill -9 -f 'claude_canon_all_v1.shard' || true
sleep 2

host=ascend.osc.edu
n=8
if ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=20 \
     -o StrictHostKeyChecking=accept-new <osc_user>@sftp.osc.edu \
     'test -d /fs/scratch/<osc_project>/bdc/data && echo SCRATCH_OK'; then
  host=sftp.osc.edu
  n=6
  echo "USE_SFTP $(date -Is)" >> "$log"
else
  echo "SFTP_FAILED $(date -Is)" >> "$log"
fi

mkdir -p "$STAGE"
rm -f "$STAGE/$ds.shard."*
( cd "$D/$ds" && find . \( -type f -o -type l \) ! -path '*/__pycache__/*' | sed 's|^\./||' | awk -v n="$n" -v p="$STAGE/$ds.shard" '{fn=sprintf("%s.%d",p,NR%n); print >> fn; close(fn)}' )
for i in $(seq 0 $((n - 1))); do
  setsid nohup bash ./upload_shard.sh "$ds" "$STAGE/$ds.shard.$i" "$i" "$host" </dev/null >/dev/null 2>&1 &
  sleep 2
done
echo "RESTARTED $host streams $n $(date -Is)" >> "$log"
