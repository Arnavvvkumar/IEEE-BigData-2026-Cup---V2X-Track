#!/bin/bash
# Measure total upload rate leaving WSL over 30 s (covers the Globus transfer), then a small extra upload test.
a=$(cat /sys/class/net/eth0/statistics/tx_bytes); sleep 30; b=$(cat /sys/class/net/eth0/statistics/tx_bytes)
echo "WSL upload over 30 s: $(( (b - a) / 30 / 1024 )) KB/s"
head -c 2000000 /dev/urandom > /tmp/up.bin
timeout 60 curl -s -o /dev/null -w "extra upload test: %{speed_upload} B/s in %{time_total}s\n" --data-binary @/tmp/up.bin https://speed.cloudflare.com/__up || echo "extra upload test did not finish in 60 s"
rm -f /tmp/up.bin
~/gcp/globusconnectpersonal -status 2>/dev/null | tr '\n' ' '; echo
