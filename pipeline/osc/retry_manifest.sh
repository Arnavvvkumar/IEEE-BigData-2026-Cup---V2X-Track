#!/bin/bash
set -uo pipefail
ds=claude_twin_v1
R=/fs/scratch/<osc_project>/bdc/data
raw="${BDC_OUT}/upload_stage/$ds.remote.raw"
mkdir -p "$(dirname "$raw")"
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu \
  "mkdir -p '$R/$ds' && cd '$R/$ds' && echo COUNT \$(find . -type f -printf '%s\t%P\n' | tee /tmp/bdc_manifest_$ds | wc -l) && cat /tmp/bdc_manifest_$ds" \
  > "$raw"
echo "ssh_exit $?"
wc -l "$raw"
head -n 5 "$raw"
echo "..."
tail -n 3 "$raw"
