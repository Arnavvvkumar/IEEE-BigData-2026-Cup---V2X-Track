#!/bin/bash
# Snapshot of the live upload: wire rate, rsync CPU, and whether zstd is available.
set -uo pipefail
echo "=== rsync versions ==="
rsync --version | head -n 2
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu 'rsync --version | head -n 2' 2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
echo "=== compress choices ==="
rsync --help 2>/dev/null | grep -E -i 'compress-choice|zstd' || echo "no local zstd choice"
echo "=== process ==="
ps -o pid,pcpu,pmem,etime,cmd -p 4193,4195 2>/dev/null || ps -o pid,pcpu,pmem,etime,cmd -C rsync
echo "=== wire rate 15s ==="
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 15
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
echo "sent_KB_per_s $(( (b - a) / 15 / 1000 ))"
echo "=== cpu again ==="
ps -o pid,pcpu,pmem,etime,cmd -C rsync -C ssh | head -n 20
