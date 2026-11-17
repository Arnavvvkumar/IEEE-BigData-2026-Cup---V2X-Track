#!/bin/bash
cd ~/.globusonline/lta 2>/dev/null || exit 1
ls
echo "--- status"; ~/gcp/globusconnectpersonal -status
echo "--- procs"; pgrep -af 'globus|gc|gridftp' | cut -c1-160
echo "--- gridftp.log tail"; tail -n 25 gridftp.log 2>/dev/null | cut -c1-220
echo "--- gridftp_*.log tail"; for f in gridftp_*.log; do [ -f "$f" ] && { echo "[$f]"; tail -n 12 "$f" | cut -c1-220; }; done
echo "--- rsync state (should be T = stopped)"; ps -o pid,stat,cmd -C rsync | head -n 4 | cut -c1-100
echo "--- relay port reachability"
for h in relay.globusonline.org; do getent hosts $h | head -n 2; done
timeout 8 bash -c 'exec 3<>/dev/tcp/relay.globusonline.org/2223' 2>&1 && echo "2223 open" || echo "2223 blocked/timeout"
