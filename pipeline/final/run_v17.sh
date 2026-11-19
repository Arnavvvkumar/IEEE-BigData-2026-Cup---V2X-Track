#!/bin/bash
# v17 = v16 + pedestrian min-points rule (boxes with < 5 scan points inside -> score x 0.25). v17s = v13 vehicle/truck + v17 pedestrian.
# Builds locally only; nothing is submitted.
F="${BDC_ROOT}/experiments/claude_carla_overnight/final"
C="$F/../coop"
P="${BDC_PYTHON}"
sed -i 's/\r$//' "$F/build_candidate_v7.sh" "$C/verify_minpts.py"
(cd "$C" && "$P" verify_minpts.py dev150 px_base_norule && "$P" verify_minpts.py dev350 px_base_norule)
export FUSE_YAW_CIRC_EXTRA=pedestrian PED_MINPTS=1
EXTRA="pvrcnn_ped_P4_ep5:pedestrian" bash "$F/run_v14.sh" 17
bash "$F/run_splice.sh" 17 17s
cd ${BDC_OUT}
python3 - <<'EOF'
import zipfile, json
a = json.loads(zipfile.ZipFile('bdc2026_trackB_submission_v16s.zip').read('predictions.json'))
b = json.loads(zipfile.ZipFile('bdc2026_trackB_submission_v17s.zip').read('predictions.json'))
def items(p, c):
    return {f: [x for x in (v if isinstance(v, list) else v.get('detections', v.get('objects', []))) if x.get('class', x.get('label')) == c] for f, v in p.items()}
for c in ('vehicle', 'truck'):
    print(c, 'identical to v16s:', items(a, c) == items(b, c))
pa, pb = items(a, 'pedestrian'), items(b, 'pedestrian')
ka = sorted((tuple(round(v, 5) for v in x['box']) for f in pa for x in pa[f]))
kb = sorted((tuple(round(v, 5) for v in x['box']) for f in pb for x in pb[f]))
print('pedestrian counts', len(ka), len(kb), '| same set of boxes (geometry):', ka == kb)
sa = sorted(x['score'] for f in pa for x in pa[f]); sb = sorted(x['score'] for f in pb for x in pb[f])
print('pedestrian score sum v16s', round(sum(sa), 2), 'v17s', round(sum(sb), 2))
EOF
echo V17_ALL_DONE
