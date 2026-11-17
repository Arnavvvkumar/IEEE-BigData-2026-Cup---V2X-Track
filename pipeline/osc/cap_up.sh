#!/bin/bash
# Cap WSL egress so the router's upload queue stays empty (bufferbloat: rtt 2-6 s, 27% retransmits).
# usage: cap_up.sh <rate e.g. 5mbit>   |   cap_up.sh off   (removes the cap; also gone after wsl --shutdown)
if [ "${1:-}" = "off" ]; then sudo -n tc qdisc del dev eth0 root 2>&1 && echo "cap removed"; exit 0; fi
sudo -n tc qdisc del dev eth0 root 2>/dev/null
sudo -n tc qdisc add dev eth0 root tbf rate "${1:-5mbit}" burst 16kb latency 100ms 2>&1 || { echo "tc failed (needs sudo/tc)"; exit 1; }
tc qdisc show dev eth0
