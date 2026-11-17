#!/bin/bash
echo "=== ifaces ==="
awk '/:/ && !/lo/ {print $1}' /proc/net/dev
echo "=== parents ==="
ps -eo pid,ppid,etime,cmd | grep -E 'upload_six|upload_shard.sh claude_canon' | grep -v grep
echo "=== rate ==="
iface=$(awk '/eth0:/ {print "eth0"; exit} /enp/ {split($1,a,":"); print a[1]; exit}' /proc/net/dev)
echo "iface $iface"
a=$(awk -v i="$iface" -F'[: ]+' -v n="$iface" '$1==n || $2==n {print $11}' /proc/net/dev)
sleep 10
b=$(awk -v i="$iface" -F'[: ]+' -v n="$iface" '$1==n || $2==n {print $11}' /proc/net/dev)
python3 -c "print('upload_Mbps', round(($b-$a)/10*8/1e6, 2))"
