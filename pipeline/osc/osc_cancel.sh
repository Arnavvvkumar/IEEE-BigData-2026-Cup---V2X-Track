#!/bin/bash
# Cancel every job of this user on OSC (at the user's explicit request) and show that the queue is empty.
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
echo "before:"; squeue -u ${OSC_USER}
scancel -u ${OSC_USER}
sleep 8
echo "after:"; squeue -u ${OSC_USER}
sacct -u ${OSC_USER} -S today -X --format=JobID,JobName%14,Partition%14,State%14,Elapsed,AllocTRES%40 2>/dev/null | tail -n 6
EOF
