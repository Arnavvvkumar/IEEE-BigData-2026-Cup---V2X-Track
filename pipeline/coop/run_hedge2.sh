#!/bin/bash
# Hedging round 2 (CPU): waits for round 1, then pedestrians, trucks, vehicles on dev150, then the same on dev350.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
while pgrep -f ped_hedge_test.py > /dev/null; do sleep 15; done
for s in dev150 dev350; do
  for c in pedestrian truck vehicle; do
    echo "=== $s $c $(date -Is)"
    "$P" hedge2.py $s v6_norule $c 4
  done
done
echo HEDGE2_ALL_DONE
