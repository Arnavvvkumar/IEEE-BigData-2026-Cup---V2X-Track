#!/bin/bash
# Why the upload is slow: which interface, remote login-node load, TCP retransmits.
set -uo pipefail
echo "=== ifaces before ==="
awk 'NR>2 {print $1, $2, $10}' /proc/net/dev
echo "=== remote load and compile ==="
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu \
  'uptime; ps -eo pcpu,pmem,comm --sort=-pcpu | head -n 8' \
  2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
echo "=== tcp to the rsync ssh ==="
ss -tinp 2>/dev/null | grep -A 2 'ascend.osc.edu' | head -n 40 || netstat -tn 2>/dev/null | head
echo "=== 20s deltas ==="
awk 'NR>2 {gsub(/:/,"",$1); print $1, $10}' /proc/net/dev > /tmp/net_a.txt
sleep 20
awk 'NR>2 {gsub(/:/,"",$1); print $1, $10}' /proc/net/dev > /tmp/net_b.txt
echo "iface sent_KB_per_s"
paste /tmp/net_a.txt /tmp/net_b.txt | awk '{d=$4-$2; if (d<0) d=0; printf "%s %d\n", $1, d/20/1000}'
echo "=== retransmits after ==="
ss -tinp 2>/dev/null | grep -E 'ascend|retrans' | head -n 20
