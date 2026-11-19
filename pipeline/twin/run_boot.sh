#!/bin/bash
# Calibration: does the bootstrap realism check predict the official v3 > v4 result?
cd "${BDC_ROOT}/experiments/claude_carla_overnight/twin"
export PYTHONPATH="${BDC_REALISM_DEPS}"
P="${BDC_PYTHON}"
"$P" realism_boot.py final_synthetic_v3/synthetic final_synthetic_v4/synthetic 60 3 all
