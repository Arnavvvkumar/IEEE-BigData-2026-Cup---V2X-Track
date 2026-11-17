#!/bin/bash
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes <osc_user>@ascend.osc.edu 'bash -s' 2>/dev/null << 'EOF' | cut -c1-200
R=/fs/scratch/<osc_project>/bdc
squeue -u <osc_user> | tail -n 2
tail -n 12 $R/logs/smoke_8533802.out
tail -n 5 $R/logs/smoke_8533802.err
EOF
