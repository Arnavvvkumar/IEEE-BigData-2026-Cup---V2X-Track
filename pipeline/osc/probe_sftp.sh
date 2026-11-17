#!/bin/bash
# READ-ONLY reachability probes (no data transferred, upload untouched)
for h in sftp.osc.edu ascend.osc.edu transfer.osc.edu; do
  for port in 22; do
    if timeout 12 bash -c "exec 3<>/dev/tcp/$h/$port" 2>/dev/null; then echo "$h:$port OPEN"; else echo "$h:$port no connect (timeout/refused)"; fi
  done
done
echo "--- sftp.osc.edu ssh banner test"
timeout 25 ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=15 <osc_user>@sftp.osc.edu 'echo hello' 2>&1 | grep -v -E '^\*|authorized|computer|authority|subject to|recorded|activities|expressly|improperly|monitoring|personnel|officials' | head -n 3
echo "--- local ssh config / tools"
which globusconnectpersonal globus 2>/dev/null || echo "no globus cli in WSL"
