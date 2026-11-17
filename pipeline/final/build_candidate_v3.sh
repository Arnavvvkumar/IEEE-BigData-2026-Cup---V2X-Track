#!/bin/bash
# Candidate v3 test predictions: j_ttaBothPed fusion, calibration, then the truck size rule.
# Requires the two pedestrian TTA files in test_allpred/. Does not submit anything.
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/final"
P="$HOME/projects/detector-program/env310/bin/python"
T="$here/test_allpred"
scans="$root/starting_kit/track_b/detection_test_frames"
out="$here/candidate_v3"
meta="$here/candidate_v3_meta"
mkdir -p "$out" "$meta"
for f in \
  vnx010_all_s1234_ep10_allpred.json \
  vnx010_all_rs2_amp_ep12_allpred.json \
  vnx010_twin_s1_ep12_allpred.json \
  canon_all_s1234_ep12_allpred.json \
  second_twin_s1_ep12_allpred.json \
  canon_relabel_s1234_ep12_allpred.json \
  vnx010_twin_s1_tta_ep12_allpred.json \
  vnx010_all_s1234_tta_ep10_allpred.json
do
  [ -f "$T/$f" ] || { echo "missing $T/$f"; exit 1; }
done
"$P" "$here/fuse_test.py" "$scans" "$meta/fused.json" \
  "$T/vnx010_all_s1234_ep10_allpred.json::pedestrian,vehicle" \
  "$T/vnx010_all_rs2_amp_ep12_allpred.json::pedestrian" \
  "$T/vnx010_twin_s1_ep12_allpred.json::pedestrian,vehicle" \
  "$T/canon_all_s1234_ep12_allpred.json::vehicle,truck" \
  "$T/second_twin_s1_ep12_allpred.json::vehicle,truck" \
  "$T/canon_relabel_s1234_ep12_allpred.json::vehicle,truck" \
  "$T/vnx010_twin_s1_tta_ep12_allpred.json::pedestrian" \
  "$T/vnx010_all_s1234_tta_ep10_allpred.json::pedestrian"
"$P" "$here/calib.py" "$scans" "$meta/fused.json" "$out/predictions.json"
rm -rf "$out/synthetic"
mkdir -p "$out/synthetic"
cp "$here/candidate_v2/synthetic/"*.bin "$out/synthetic/"
"$P" "$here/check_candidate.py" "$out/predictions.json"
echo "synthetic $(ls "$out/synthetic" | wc -l)"
echo CANDIDATE_V3_DONE
