#!/bin/bash
# Candidate v4 test predictions = the "o_truck" ensemble (dev150 69.43 / dev350 68.73 against v3 66.63 / 65.4):
# v3 + second_twin_s2 + vnx010_twin_s2 (pedestrian) + small-truck SECOND (truck) + size-aware VoxelNeXt (vehicle) + full-set SECOND (truck),
# averaged vehicle heading in fusion, calibration, truck size rule, truck de-duplication. Synthetic folder unchanged from v2/v3.
# Does not submit anything.
set -euo pipefail
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
P="${BDC_PYTHON}"
T="$here/test_allpred"
scans="$root/starting_kit/track_b/detection_test_frames"
out="$here/candidate_v4"
meta="$here/candidate_v4_meta"
mkdir -p "$out" "$meta"
SPEC="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian vnx010_twin_s2_ep12:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck"
args=()
for s in $SPEC; do
  f="$T/${s%%:*}_allpred.json"
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  args+=("$f::${s##*:}")
done
FUSE_YAW_CIRC=vehicle "$P" "$here/fuse_test.py" "$scans" "$meta/fused.json" "${args[@]}"
"$P" "$here/calib.py" "$scans" "$meta/fused.json" "$meta/calibrated.json"
cd "$here"
"$P" - "$meta/calibrated.json" "$out/predictions.json" << 'PY'
import json, sys
from pathlib import Path
import calib
pred = json.loads(Path(sys.argv[1]).read_text())
n0 = sum(d['class'] == 'truck' for v in pred.values() for d in v)
pred = calib.truck_dedupe(pred)
n1 = sum(d['class'] == 'truck' for v in pred.values() for d in v)
Path(sys.argv[2]).write_text(json.dumps(pred))
print('truck_dedupe', n0, '->', n1, 'truck boxes;', len(pred), 'frames')
PY
rm -rf "$out/synthetic"
mkdir -p "$out/synthetic"
cp "$here/candidate_v3/synthetic/"*.bin "$out/synthetic/"
"$P" "$here/check_candidate.py" "$out/predictions.json"
echo "synthetic $(ls "$out/synthetic" | wc -l)"
bash "$here/finish_zip.sh" candidate_v4 bdc2026_trackB_submission_v4.zip
"$P" "$here/verify_zip.py" "${BDC_OUT}/bdc2026_trackB_submission_v4.zip"
echo CANDIDATE_V4_DONE
