#!/bin/bash
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop"
P="$HOME/projects/detector-program/env310/bin/python"
for s in dev150 dev350; do "$P" ped_yaw_hedge.py $s pf_yawcirc_norule 3; done
