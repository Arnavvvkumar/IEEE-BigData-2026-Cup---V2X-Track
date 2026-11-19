#!/bin/bash
# Vehicle and truck box calibration tests on the dev frames (CPU).
cd "${BDC_ROOT}/experiments/claude_carla_overnight/erroranalysis"
P="${BDC_PYTHON}"
for c in vehicle truck; do
  nice -n 5 "$P" class_adjust.py ../twin/static_filter_cw_d_predictions.json "class_adjust_$c.json" "$c"
done
echo CLASS_ADJUST_DONE
