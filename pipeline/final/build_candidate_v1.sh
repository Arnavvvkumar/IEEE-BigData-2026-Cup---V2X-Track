#!/bin/bash
# Candidate v1 test predictions: ensemble fuse_cw_d (dev 64.48 with pedestrian static down-weighting).
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/final"
P="$HOME/projects/bdc2026-track-b/.env/bin/python"
T="$here/test_allpred"
VT="vehicle,truck"
cd "$here"
"$P" fuse_test.py "$root/starting_kit/track_b/detection_test_frames" "$here/candidate_v1/predictions.json" \
  "$T/vnx010_all_s1234_ep10_allpred.json::pedestrian,vehicle" "$T/vnx010_all_rs2_amp_ep12_allpred.json::pedestrian" \
  "$T/canon_all_s1234_ep12_allpred.json::$VT" "$T/second_twin_s1_ep12_allpred.json::$VT" "$T/canon_relabel_s1234_ep12_allpred.json::$VT"
"$P" check_candidate.py "$here/candidate_v1/predictions.json"
