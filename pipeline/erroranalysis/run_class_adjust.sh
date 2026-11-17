#!/bin/bash
# Vehicle and truck box calibration tests on the dev frames (CPU).
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/erroranalysis"
P="$HOME/projects/detector-program/env310/bin/python"
for c in vehicle truck; do
  nice -n 5 "$P" class_adjust.py ../twin/static_filter_cw_d_predictions.json "class_adjust_$c.json" "$c"
done
echo CLASS_ADJUST_DONE
