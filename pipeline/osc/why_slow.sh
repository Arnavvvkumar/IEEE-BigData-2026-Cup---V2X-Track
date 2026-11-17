#!/bin/bash
echo "=== cpu ==="
ps -eo pcpu,cmd | grep -E 'rsync -aL|ssh -i' | grep -v grep | awk '{s+=$1} END {print "cpu_pct", s+0}'
echo "=== tcp ==="
ss -tinp 2>/dev/null | awk '
  /ascend.osc.edu|192.148.247/ {p=1}
  p && /cwnd|rtt|bytes_acked|retrans|rwnd|send / {print; p=0}
' | head -n 40
echo "=== ss count ==="
ss -tnp 2>/dev/null | grep -c '192.148.247' || true
echo "=== rate ==="
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 8
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
python3 -c "print(round(($b-$a)/8*8/1e6, 2))"
