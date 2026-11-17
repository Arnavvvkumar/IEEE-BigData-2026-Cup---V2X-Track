#!/bin/bash
# Stronger truck size hedging (variants at 20-40% of the score) on the v9 bundle-B ensemble, both validation sets.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
for s in dev150 dev350; do
  echo "=== $s $(date -Is)"
  "$P" hedge2.py $s v9B_norule truck2 4
done
echo TRUCK2_ALL_DONE
