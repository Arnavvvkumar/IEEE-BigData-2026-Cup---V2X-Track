#!/bin/bash
# Candidate v7 = v6 (v5 vehicles/trucks + v3 pedestrian voters) + long-low-truck SECOND as truck voter + fourth VoxelNeXt fine-tune as
# vehicle voter + pedestrian box hedging (calib.hedge_pedestrians). Truck rule a 0.5 base 0.1, truck de-duplication, averaged vehicle
# heading. Synthetic folder unchanged. Does not submit anything.
# usage: build_candidate_v7.sh [name=candidate_v7] [zip=bdc2026_trackB_submission_v7.zip] [hedge=1]
set -euo pipefail
name=${1:-candidate_v7}; zip=${2:-bdc2026_trackB_submission_v7.zip}; export HEDGE=${3:-1}
root="${BDC_ROOT}"
here="$root/experiments/claude_carla_overnight/final"
P="${BDC_PYTHON}"
T="${TESTPRED:-$here/test_allpred}"   # TESTPRED=$here/test_allpred_ps for the per-scan levelled (compliant) exports
scans="$root/starting_kit/track_b/detection_test_frames"
out="$here/$name"; meta="$here/${name}_meta"
mkdir -p "$out" "$meta"
SPEC="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle"
SPEC="$SPEC ${EXTRA:-}"   # optional extra passes, e.g. EXTRA="pvrcnn_st_osc_ep2:pedestrian,vehicle"
if [ -n "${DROP:-}" ]; then   # DROP="second_twin_ll_s1_ep12" removes passes whose model name matches exactly
  keep=""; for s in $SPEC; do [[ " $DROP " == *" ${s%%:*} "* ]] || keep="$keep $s"; done; SPEC="$keep"
fi
echo "spec: $SPEC"
echo "test predictions from: $T"
args=()
for s in $SPEC; do
  f="$T/${s%%:*}_allpred.json"
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  args+=("$f::${s##*:}")
done
FUSE_YAW_CIRC=vehicle "$P" "$here/fuse_test.py" "$scans" "$meta/fused.json" "${args[@]}"
cd "$here"
"$P" - "$scans" "$meta/fused.json" "$out/predictions.json" << 'PY'
import json, os, sys
from pathlib import Path
import numpy as np
import calib
clouds = {f.stem: np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted(Path(sys.argv[1]).glob('*.bin'))}
pred = calib.calibrate(json.loads(Path(sys.argv[2]).read_text()), clouds)
rule = {**calib.TRUCK, 'a': float(os.environ.get('TRUCK_A', '0.5')), 'base': float(os.environ.get('TRUCK_BASE', '0.1'))}
pred = calib.truck_dedupe(calib.truck_rule(pred, **rule))
if os.environ.get('TRUCK_MINPTS', '0') == '1':   # v18: truck boxes with < 30 scan points inside -> score x 0.25 (before the truck size hedge)
    pred = calib.truck_min_points(pred, clouds)
cnt = lambda p: {c: sum(d['class'] == c for v in p.values() for d in v) for c in ('vehicle', 'pedestrian', 'truck')}
before = cnt(pred)
if os.environ.get('TRUCKPP', '0') == '1':
    pred = calib.truck_boost_hedge(pred, clouds, boost=float(os.environ.get('TRUCKBOOST', '2.0')))
if os.environ.get('PED_MINPTS', '0') == '1':   # v17: pedestrian boxes with < 5 scan points inside -> score x 0.25
    pred = calib.ped_min_points(pred, clouds)
if os.environ.get('HEDGE', '1') == '1':
    pred = calib.hedge_pedestrians(pred)
Path(sys.argv[3]).write_text(json.dumps(pred))
print('rule', rule, '| boxes before hedging', before, '| after', cnt(pred), '|', len(pred), 'frames | hedge', os.environ.get('HEDGE'))
PY
rm -rf "$out/synthetic"; mkdir -p "$out/synthetic"
cp "${SYNDIR:-$here/candidate_v3/synthetic}/"*.bin "$out/synthetic/"
echo "synthetic from ${SYNDIR:-$here/candidate_v3/synthetic}"
"$P" "$here/check_candidate.py" "$out/predictions.json" | head -n 2
echo "synthetic $(ls "$out/synthetic" | wc -l)"
bash "$here/finish_zip.sh" "$name" "$zip" 2>/dev/null || true
"$P" "$here/verify_zip.py" "${BDC_OUT}/$zip" | tail -n 2
"$P" - "$here" "$name" << 'PY'
import json, sys
from pathlib import Path
h = Path(sys.argv[1])
a = json.loads((h / 'candidate_v6/predictions.json').read_text()); b = json.loads((h / sys.argv[2] / 'predictions.json').read_text())
key = lambda x: tuple(round(v, 6) for v in x['box'])
pa = {f: {key(x) for x in a[f] if x['class'] == 'pedestrian'} for f in a}
pb = {f: {key(x) for x in b[f] if x['class'] == 'pedestrian'} for f in b}
print('every v6 pedestrian box still present in', sys.argv[2], ':', all(pa[f] <= pb[f] for f in a))
mb = sum((h / sys.argv[2] / p).stat().st_size for p in ['predictions.json']) / 1e6 + sum(q.stat().st_size for q in (h / sys.argv[2] / 'synthetic').glob('*.bin')) / 1e6
print(f'uncompressed size {mb:.1f} MB (limit about 75 MB)')
PY
echo CANDIDATE_BUILT "$name"
