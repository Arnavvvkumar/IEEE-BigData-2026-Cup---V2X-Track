#!/bin/bash
cd ~/gcp
./globusconnectpersonal -stop; sleep 5
pkill -f 'gc.py|gc-ctrl.py' 2>/dev/null; sleep 2
setsid nohup ./globusconnectpersonal -start > ~/gcp_start.log 2>&1 < /dev/null &
sleep 35
./globusconnectpersonal -status
echo "--- relay tunnel"; ss -tnp 2>/dev/null | grep -E 'pid=[0-9]+' | grep -E 'ssh' | grep -v -E '192\.148\.247' | cut -c1-150 | head -n 4
echo "--- gridftp procs"; pgrep -af 'gridftp|gfork' | cut -c1-120 | head -n 3
