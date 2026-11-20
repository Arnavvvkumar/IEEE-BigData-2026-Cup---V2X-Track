#!/bin/bash
# Read-only ETA. Does not stop the 12 canon rsyncs.
echo "=== rate ==="
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 12
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
python3 -c "print(round(($b-$a)/12*8/1e6, 3))" > /tmp/osc_mbps
echo "upload_Mbps $(cat /tmp/osc_mbps)"
echo "rsyncs $(ps -eo cmd | grep 'rsync -aL' | grep -c 'claude_canon_all_v1.shard')"
echo "local_canon_MB $(du -sm "${BDC_DATA}/claude_canon_all_v1" | awk '{print $1}')"
echo "=== remote ==="
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=12 ${OSC_USER}@ascend.osc.edu \
  'du -sm /fs/scratch/${OSC_PROJECT}/bdc/data/claude_canon_all_v1' || echo "ssh_rc $?"
