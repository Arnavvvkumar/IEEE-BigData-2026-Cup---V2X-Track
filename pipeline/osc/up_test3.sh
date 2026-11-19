#!/bin/bash
# WSL-side upload test plus network details (MTU, offload settings, retransmits).
head -c 4000000 /dev/urandom > /tmp/up.bin
timeout 60 curl -s -o /dev/null -w "WSL upload: %{speed_upload} B/s in %{time_total}s\n" --data-binary @/tmp/up.bin https://speed.cloudflare.com/__up || echo "WSL upload did not finish in 60 s"
rm -f /tmp/up.bin
echo "mtu $(cat /sys/class/net/eth0/mtu)"
ethtool -k eth0 2>/dev/null | grep -E 'tcp-segmentation-offload|generic-segmentation-offload|tx-checksumming' || echo "ethtool not available"
nstat -az 2>/dev/null | grep -E 'TcpRetransSegs|TcpOutSegs' || grep -E '^Tcp:' /proc/net/snmp | cut -c1-200
grep -E 'networkingMode|\[wsl2\]' /mnt/c/Users/$USER/.wslconfig 2>/dev/null || echo "no .wslconfig networking setting"
sysctl net.ipv4.tcp_congestion_control 2>/dev/null
