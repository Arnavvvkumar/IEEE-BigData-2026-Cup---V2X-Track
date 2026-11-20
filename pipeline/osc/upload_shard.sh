#!/bin/bash
# One slice of a dataset. Retries so a dropped ssh does not end the shard.
set -uo pipefail
ds=$1
list=$2
id=$3
host=${4:-ascend.osc.edu}
here="$(cd "$(dirname "$0")" && pwd)"
D="${BDC_DATA}"
R=/fs/scratch/${OSC_PROJECT}/bdc/data
LOG="$here/upload_shard_${ds}_${id}.log"
SSHCMD="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -c aes128-gcm@openssh.com -o IPQoS=throughput"
exec >> "$LOG" 2>&1
echo "SHARD_START $ds $id $(date -Is) files $(wc -l < "$list")"
ok=0
for try in $(seq 1 20); do
  echo "try $try $(date -Is)"
  rsync -aL --partial --size-only --files-from="$list" -z --compress-level=1 --info=stats1 \
    -e "$SSHCMD" \
    "$D/$ds/" "${OSC_USER}@${host}:$R/$ds/"
  rc=$?
  echo "rsync_exit $rc $(date -Is)"
  if [ "$rc" = "0" ]; then
    ok=1
    break
  fi
  sleep 8
done
if [ "$ok" = "1" ]; then
  echo "SHARD_DONE $ds $id $(date -Is)"
else
  echo "SHARD_FAILED $ds $id $(date -Is)"
  exit 1
fi
