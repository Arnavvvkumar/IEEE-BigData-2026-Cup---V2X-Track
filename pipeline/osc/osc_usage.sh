#!/bin/bash
# READ-ONLY: OSC usage and charges for project ${OSC_PROJECT} (no job is submitted or changed).
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
echo "--- OSCusage (this month)"; OSCusage 2>&1 | head -n 40
echo "--- our jobs since Nov 11"
sacct -u ${OSC_USER} -S 2026-11-11 -X -n -P --format=JobID,JobName,Partition,State,Elapsed,AllocTRES 2>/dev/null | head -n 20
EOF
