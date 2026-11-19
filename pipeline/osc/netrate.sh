#!/bin/bash
# Outgoing network rate of this WSL VM over 20 s, plus upload job state.
cd "${BDC_ROOT}/experiments/claude_carla_overnight/osc"
cat upload_canon.log upload_twin.log 2>/dev/null | cut -c1-160
echo "tar jobs: $(pgrep -fc 'tar -C')"
a=$(awk '/eth0/ {print $10}' /proc/net/dev)
sleep 20
b=$(awk '/eth0/ {print $10}' /proc/net/dev)
echo "upload rate: $(( (b - a) / 20 / 1000 )) KB/s  (total sent so far: $(( b / 1000000 )) MB)"
