#!/bin/bash
# Pedestrian hedge re-tune on the v16 base (averaged pedestrian heading); same pre-registered grid as ped_retune.py.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
for s in dev150 dev350; do "$P" ped_retune.py $s pf_yawcirc_norule 2; done
