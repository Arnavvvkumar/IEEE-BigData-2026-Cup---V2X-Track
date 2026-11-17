#!/bin/bash
# Calibration: does the bootstrap realism check predict the official v3 > v4 result?
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/twin"
export PYTHONPATH="$HOME/projects/roadside-pilot/realism_deps"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
"$P" realism_boot.py final_synthetic_v3/synthetic final_synthetic_v4/synthetic 60 3 all
