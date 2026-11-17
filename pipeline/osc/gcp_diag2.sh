#!/bin/bash
cd ~/gcp
echo "--- logs anywhere"; find ~/.globusonline ~/gcp -name '*.log' -newer ~/gcp/LICENSE 2>/dev/null | head; cat ~/gcp_start.log | tail -n 5
echo "--- gridftp binary + missing libs"
ls etc/gridftp.d 2>/dev/null | head -n 3
GF=$(find ~/gcp -name 'globus-gridftp-server*' -type f | head -n 3); echo "$GF"
for b in $GF; do ldd "$b" 2>&1 | grep -i 'not found' | head -n 5; done
echo "--- gc.py stderr lines"; pgrep -af 'gc.py' | head -n 2 | cut -c1-120
echo "--- ssh tunnel to relay (port 2223 forward)"; ss -tnp 2>/dev/null | grep -E '2223|ssh' | grep -v osc.edu | head -n 5 | cut -c1-160
echo "--- ip6 vs ip4: relay over v4"; timeout 8 curl -4 -s -o /dev/null -w "v4 https relay: %{http_code}\n" https://relay.globusonline.org:2223/ || echo v4 https probe failed
