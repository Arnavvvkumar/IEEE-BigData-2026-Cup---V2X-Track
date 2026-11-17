#!/bin/bash
# Point-extent pedestrian size rule: grid on dev350 then dev150 (CPU only).
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
for s in dev350 dev150; do
  echo "=== $s $(date -Is)"
  "$P" ped_size_rule.py $s st_truckonly
done
