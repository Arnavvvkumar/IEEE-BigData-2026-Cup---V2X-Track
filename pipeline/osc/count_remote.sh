#!/bin/bash
# Read-only. Count files already on OSC. Does not start or stop rsync.
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=20 ${OSC_USER}@ascend.osc.edu \
  'echo twin $(ls /fs/scratch/${OSC_PROJECT}/bdc/data/claude_twin_v1/points | wc -l); echo canon $(ls /fs/scratch/${OSC_PROJECT}/bdc/data/claude_canon_all_v1/points | wc -l)' \
  2>/dev/null | grep -E '^(twin|canon) '
