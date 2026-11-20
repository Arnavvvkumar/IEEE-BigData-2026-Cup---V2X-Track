#!/bin/bash
# Resumable compressed upload of the synthetic training sets to OSC scratch (rsync -z, symlinks dereferenced, retries on drops).
# Synthetic data only. usage: upload_rsync.sh  (runs twin first, then canonical)
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
D="${BDC_DATA}"
R=/fs/scratch/${OSC_PROJECT}/bdc/data
SSHCMD="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -c aes128-gcm@openssh.com"
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes ${OSC_USER}@ascend.osc.edu "mkdir -p $R" 2>/dev/null
for ds in claude_twin_v1 claude_canon_all_v1; do
  for try in $(seq 1 25); do
    echo "[$(date +%T)] $ds attempt $try"
    rsync -aL --partial --inplace -z --compress-level=1 --info=stats1 -e "$SSHCMD" "$D/$ds/" "${OSC_USER}@ascend.osc.edu:$R/$ds/" 2>&1 | grep -v -E '^\*|^\s*$|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials'
    rc=${PIPESTATUS[0]}
    echo "[$(date +%T)] $ds rsync exit $rc"
    [ "$rc" = "0" ] && break
    sleep 20
  done
done
echo "UPLOAD_ALL_DONE $(date +%T)"
