#!/bin/bash
# Rebuild predictions.json of the final submission 1054222 (v18s) from the raw per-model test predictions, on the CPU,
# and compare it with submission/predictions.json.
#   source pipeline/env.sh
#   RAW=/path/to/raw_test_predictions bash pipeline/final/reproduce_v18s.sh
# RAW   : the 22 <model>_allpred.json files from release asset raw_test_predictions.zip (per-scan levelled, 6 ground heights)
# SCANS : the 50 released test scans (default: the organizers' starting kit)
# The static-structure map is read from pipeline/final/static_map.npz, so the unlabelled development clouds are not needed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RAW="${RAW:-$BDC_ROOT/raw_test_predictions}"
SCANS="${SCANS:-$BDC_ROOT/starting_kit/track_b/detection_test_frames}"
OUT="${OUT:-$BDC_OUT/reproduced_v18s}"
PY="${BDC_PYTHON:-python3}"
export FUSE_STATIC_MAP="${FUSE_STATIC_MAP-$HERE/static_map.npz}"   # set FUSE_STATIC_MAP= (empty) to rebuild the map from the dev clouds
mkdir -p "$OUT"

# passes of the final builds (build_candidate_v7.sh SPEC without the long-low truck voter, plus EXTRA); order matters for fusion
BASE="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
COMMON="pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5)"
EXTRA_A="$COMMON"                                                                                     # build v13t: vehicle + truck boxes
EXTRA_B="$COMMON $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian" # build v17 : pedestrian boxes

build() {  # name, extra passes, then env assignments
  local name=$1 extra=$2; shift 2
  local args=()
  for s in $BASE $extra; do
    f="$RAW/${s%%:*}_allpred.json"; [ -f "$f" ] || { echo "missing $f"; exit 1; }
    args+=("$f::${s##*:}")
  done
  echo "== $name: ${#args[@]} passes"
  env "$@" FUSE_ONE_VOTE=vehicle FUSE_YAW_CIRC=vehicle "$PY" "$HERE/fuse_test.py" "$SCANS" "$OUT/fused_$name.json" "${args[@]}"
  env "$@" TRUCKPP=1 TRUCKBOOST=1.0 HEDGE=1 "$PY" "$HERE/postprocess.py" "$SCANS" "$OUT/fused_$name.json" "$OUT/pred_$name.json"
}
build v13t "$EXTRA_A" TRUCK_MINPTS=1
build v17  "$EXTRA_B" FUSE_YAW_CIRC_EXTRA=pedestrian PED_MINPTS=1

"$PY" - "$OUT" "$REPO/submission/predictions.json" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
a = json.loads((out / 'pred_v13t.json').read_text()); b = json.loads((out / 'pred_v17.json').read_text())
final = {f: [d for d in a[f] if d['class'] != 'pedestrian'] + [d for d in b[f] if d['class'] == 'pedestrian'] for f in a}
(out / 'predictions.json').write_text(json.dumps(final))
ref = json.loads(Path(sys.argv[2]).read_text())
key = lambda d: (d['class'], tuple(round(v, 6) for v in d['box']), round(d['score'], 8))
same = set(final) == set(ref) and all(sorted(map(key, final[f])) == sorted(map(key, ref[f])) for f in ref)
cnt = {c: sum(d['class'] == c for v in final.values() for d in v) for c in ('vehicle', 'pedestrian', 'truck')}
print('reproduced', cnt, '| identical to submission/predictions.json:', same)
sys.exit(0 if same else 1)
EOF
