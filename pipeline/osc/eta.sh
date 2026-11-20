#!/bin/bash
# Remaining-upload estimate from the live wire rate and scratch sizes.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bash "$here/parallel_rate.sh"
echo "=== remote ==="
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=25 ${OSC_USER}@ascend.osc.edu \
  'du -sm /fs/scratch/${OSC_PROJECT}/bdc/data /fs/scratch/${OSC_PROJECT}/bdc/data/* 2>/dev/null' \
  2>/dev/null | grep -E '^[0-9]'
