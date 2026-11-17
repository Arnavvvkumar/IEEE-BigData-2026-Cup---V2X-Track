#!/bin/bash
# Sample the upload rate leaving WSL (the Globus transfer) four times, 15 s each, plus the retransmit share.
r0=$(nstat -az 2>/dev/null | awk '/TcpRetransSegs/{print $2}'); o0=$(nstat -az 2>/dev/null | awk '/TcpOutSegs/{print $2}')
for i in 1 2 3 4; do
  a=$(cat /sys/class/net/eth0/statistics/tx_bytes); sleep 15; b=$(cat /sys/class/net/eth0/statistics/tx_bytes)
  echo "sample $i: $(( (b - a) / 15 / 1024 )) KB/s"
done
r1=$(nstat -az 2>/dev/null | awk '/TcpRetransSegs/{print $2}'); o1=$(nstat -az 2>/dev/null | awk '/TcpOutSegs/{print $2}')
echo "segments sent $((o1 - o0)) retransmitted $((r1 - r0))"
ss -tin 2>/dev/null | grep -B1 -E 'bytes_sent:[0-9]{7,}' | grep -oE '^[a-z]+ +[A-Z-]+.*|cwnd:[0-9]+|rto:[0-9]+|retrans:[0-9/]+|send [0-9.]+[MK]?bps|rtt:[0-9./]+' | tr '\n' ' ' | cut -c1-700; echo
