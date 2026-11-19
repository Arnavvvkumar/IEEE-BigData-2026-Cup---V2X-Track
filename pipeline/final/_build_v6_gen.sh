#!/bin/bash
# Candidate v5 test predictions = v4 + second size-aware VoxelNeXt fine-tune (pedestrian, vehicle) + retuned truck rule
# (a 0.5, base 0.1; coop/rule_retune.py). Validation: dev150 69.9 / dev350 68.93 (v4 69.43 / 68.73, v3 66.63 / 65.4).
# Synthetic folder unchanged. Does not submit anything.
set -euo pipefail
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
P="${BDC_PYTHON}"
T="$here/test_allpred"
scans="$root/starting_kit/track_b/detection_test_frames"
out="$here/candidate_v6"
meta="$here/candidate_v6_meta"
mkdir -p "$out" "$meta"
SPEC="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck vnx010_pv_ft_ep5:vehicle second_twin_pv_s1_ep14:truck vnx010_pv_ft2_ep5:vehicle"
args=()
for s in $SPEC; do
  f="$T/${s%%:*}_allpred.json"
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  args+=("$f::${s##*:}")
done
FUSE_YAW_CIRC=vehicle "$P" "$here/fuse_test.py" "$scans" "$meta/fused.json" "${args[@]}"
cd "$here"
"$P" - "$scans" "$meta/fused.json" "$out/predictions.json" << 'PY'
import json, sys
from pathlib import Path
import numpy as np
import calib
clouds = {f.stem: np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(Path(sys.argv[1]).glob('*.bin'))}
pred = calib.calibrate(json.loads(Path(sys.argv[2]).read_text()), clouds)
rule = {**calib.TRUCK, 'a': 0.5, 'base': 0.1}
pred = calib.truck_rule(pred, **rule)
n0 = sum(d['class'] == 'truck' for v in pred.values() for d in v)
pred = calib.truck_dedupe(pred)
n1 = sum(d['class'] == 'truck' for v in pred.values() for d in v)
Path(sys.argv[3]).write_text(json.dumps(pred))
print('rule', rule, '| truck_dedupe', n0, '->', n1, '|', len(pred), 'frames')
PY
rm -rf "$out/synthetic"
mkdir -p "$out/synthetic"
cp "$here/candidate_v3/synthetic/"*.bin "$out/synthetic/"
"$P" "$here/check_candidate.py" "$out/predictions.json"
echo "synthetic $(ls "$out/synthetic" | wc -l)"
bash "$here/finish_zip.sh" candidate_v6 bdc2026_trackB_submission_v6.zip || true
"$P" "$here/verify_zip.py" "${BDC_OUT}/bdc2026_trackB_submission_v6.zip"
echo CANDIDATE_V6_DONE
