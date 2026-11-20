#!/bin/bash
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>/dev/null << 'EOF' | cut -c1-200
R=/fs/scratch/${OSC_PROJECT}/bdc
squeue -u ${OSC_USER} | tail -n 2
tail -n 12 $R/logs/smoke_8533802.out
tail -n 5 $R/logs/smoke_8533802.err
EOF
