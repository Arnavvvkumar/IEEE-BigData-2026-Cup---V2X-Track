#!/bin/bash
# Stream the two synthetic training sets to OSC scratch (tar over ssh, symlinks dereferenced). Synthetic data only.
# usage: upload_data.sh <dataset name>
set -uo pipefail
ds=$1
D="${BDC_DATA}"
R=/fs/scratch/<osc_project>/bdc/data
SSH="ssh -i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=30 <osc_user>@ascend.osc.edu"
$SSH "mkdir -p $R" 2>/dev/null
start=$(date +%s)
tar -C "$D" -chf - "$ds" | $SSH "tar -C $R -xf -"
echo "UPLOAD_EXIT $? $ds seconds=$(( $(date +%s) - start ))"
$SSH "echo remote files: \$(find $R/$ds -type f | wc -l); du -sh $R/$ds | cut -f1" 2>/dev/null
