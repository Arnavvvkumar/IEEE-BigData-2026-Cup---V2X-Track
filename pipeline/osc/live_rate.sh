#!/bin/bash
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 10
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
python3 -c "print('upload_MBps', round(($b-$a)/10/1e6, 3)); print('upload_Mbps', round(($b-$a)/10*8/1e6, 2))"
pgrep -af 'rsync -aL' || echo 'rsync not running'
