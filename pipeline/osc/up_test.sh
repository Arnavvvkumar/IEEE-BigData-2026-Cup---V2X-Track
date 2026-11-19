#!/bin/bash
# Measure raw upload speed (random bytes to a public speed-test endpoint) and show what the Globus share holds.
head -c 3000000 /dev/urandom > /tmp/up.bin
timeout 90 curl -s -o /dev/null -w "upload %{speed_upload} B/s in %{time_total}s (while Globus is also sending)\n" --data-binary @/tmp/up.bin https://speed.cloudflare.com/__up || echo "upload test failed"
rm -f /tmp/up.bin
cd ${BDC_DATA}
du -sh --apparent-size */ 2>/dev/null | sort -h | tail -n 15
echo "rsync: $(ps -o stat= -C rsync | sort | uniq -c | tr '\n' ' ')"
