#!/bin/bash
# Ensemble inference on the staged validation frames. usage: run_dev150_infer.sh <set name, e.g. dev150>
base="${BDC_ROOT}/experiments/claude_carla_overnight"
bash "$base/final/run_infer_dir.sh" "$base/coop/$1/scans" "$base/coop/$1/allpred"
