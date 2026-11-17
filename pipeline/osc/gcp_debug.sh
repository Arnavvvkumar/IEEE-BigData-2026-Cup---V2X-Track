#!/bin/bash
cd ~/gcp
./globusconnectpersonal -h 2>&1 | head -n 25
echo "--- openssl / distro"; openssl version; lsb_release -d 2>/dev/null
echo "--- lib dirs"; ls gt_amd64 | head; ls gt_amd64/lib 2>/dev/null | head -n 5
echo "--- gridftp server direct test (library path as the launcher sets it)"
export LD_LIBRARY_PATH="$HOME/gcp/gt_amd64/lib:$HOME/gcp/gt_amd64/lib64:${LD_LIBRARY_PATH:-}"
timeout 10 ./gt_amd64/sbin/globus-gridftp-server -version 2>&1 | head -n 5
ldd ./gt_amd64/sbin/globus-gridftp-server 2>&1 | grep -i 'not found' | head -n 5
echo "--- restart in debug and read its log"
./globusconnectpersonal -stop >/dev/null 2>&1; sleep 4
setsid nohup ./globusconnectpersonal -debug -start > ~/gcp_start.log 2>&1 < /dev/null &
sleep 30
./globusconnectpersonal -status
ls -la ~/.globusonline/lta/*.log 2>/dev/null | cut -c30-120
tail -n 20 ~/.globusonline/lta/gridftp.log 2>/dev/null | cut -c1-200
