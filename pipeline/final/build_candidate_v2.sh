#!/bin/bash
# Candidate v2 = candidate v1 ensemble (fuse_cw_d + pedestrian static down-weighting) + box calibration (final/calib.py).
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/final"
P="$HOME/projects/detector-program/env310/bin/python"
scans="$root/starting_kit/track_b/detection_test_frames"
cd "$here"
echo "== dev check"
"$P" check_calib_dev.py "$here/../twin/static_filter_cw_d_predictions.json" "$here/candidate_v2_meta/dev_predictions_calibrated.json"
echo "== test predictions"
"$P" calib.py "$scans" "$here/candidate_v1/predictions.json" "$here/candidate_v2/predictions.json"
mkdir -p candidate_v2/synthetic && cp candidate_v1/synthetic/*.bin candidate_v2/synthetic/
"$HOME/projects/bdc2026-track-b/.env/bin/python" check_candidate.py "$here/candidate_v2/predictions.json"
ls candidate_v2 && ls candidate_v2/synthetic | wc -l
echo CANDIDATE_V2_DONE
