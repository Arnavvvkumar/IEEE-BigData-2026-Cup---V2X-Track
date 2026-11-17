#!/bin/bash
for u in https://auth.globus.org/ https://app.globus.org/ https://www.google.com/ https://downloads.globus.org/; do
  timeout 25 curl -s -o /dev/null -w "$u HTTP %{http_code} connect %{time_connect}s tls %{time_appconnect}s total %{time_total}s\n" "$u" || echo "$u FAILED"
done
echo "MTU: $(cat /sys/class/net/eth0/mtu)"
