#!/bin/bash
for p in $(pgrep -f 'bash laneC\.sh'); do kill "$p"; done
sleep 1
for p in $(pgrep -f 'extra_tag pvrcnn_all_s1234'); do kill "$p"; done
sleep 3
rm -rf "${BDC_PROGRAM}/OpenPCDet/output${BDC_ROOT}/experiments/claude_carla_overnight/canon/pvrcnn_all/pvrcnn_all_s1234"
rm -rf "${BDC_PROGRAM}/OpenPCDet/output${BDC_ROOT}/experiments/claude_carla_overnight/canon/vnx010_all/vnx010_all_s1234"
pgrep -af 'extra_tag' | sed 's/.*extra_tag \([a-z0-9_]*\).*/\1/' | sort | uniq -c
free -m | sed -n 2p
