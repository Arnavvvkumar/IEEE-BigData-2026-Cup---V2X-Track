#!/bin/bash
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=15 <osc_user>@ascend.osc.edu \
  'echo twin $(ls /fs/scratch/<osc_project>/bdc/data/claude_twin_v1/points | wc -l); echo canon $(ls /fs/scratch/<osc_project>/bdc/data/claude_canon_all_v1/points | wc -l)'
echo "ssh_rc $?"
