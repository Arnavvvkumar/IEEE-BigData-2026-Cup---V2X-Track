#!/bin/bash
# Cancel every job of this user on OSC (at the user's explicit request) and show that the queue is empty.
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 <osc_user>@ascend.osc.edu 'bash -s' 2>&1 << 'EOF' | grep -v -E 'authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials|^\*|^\s*$' | cut -c1-200
source /etc/profile 2>/dev/null
echo "before:"; squeue -u <osc_user>
scancel -u <osc_user>
sleep 8
echo "after:"; squeue -u <osc_user>
sacct -u <osc_user> -S today -X --format=JobID,JobName%14,Partition%14,State%14,Elapsed,AllocTRES%40 2>/dev/null | tail -n 6
EOF
