#!/bin/bash
# v16 = v15 recipe (P4 x1 extra) + averaged (circular-mean) pedestrian heading in fusion. Then v16s = v13 vehicle/truck + v16 pedestrian.
# Builds locally only; nothing is submitted.
F="${BDC_ROOT}/experiments/claude_carla_overnight/final"
sed -i 's/\r$//' "$F/run_v14.sh" "$F/run_splice.sh"
export FUSE_YAW_CIRC_EXTRA=pedestrian
EXTRA="pvrcnn_ped_P4_ep5:pedestrian" bash "$F/run_v14.sh" 16
bash "$F/run_splice.sh" 16 16s
cd ${BDC_OUT}
python3 - <<'EOF'
import zipfile, json, math
a = json.loads(zipfile.ZipFile('bdc2026_trackB_submission_v15s.zip').read('predictions.json'))
b = json.loads(zipfile.ZipFile('bdc2026_trackB_submission_v16s.zip').read('predictions.json'))
def items(p, c):
    out = {}
    for f, v in p.items():
        v = v if isinstance(v, list) else v.get('detections', v.get('objects', []))
        out[f] = [x for x in v if x.get('class', x.get('label')) == c]
    return out
for c in ('vehicle', 'truck'):
    print(c, 'identical to v15s:', items(a, c) == items(b, c))
pa, pb = items(a, 'pedestrian'), items(b, 'pedestrian')
print('pedestrian counts', sum(map(len, pa.values())), sum(map(len, pb.values())))
same_xy = sum(1 for f in pa for x, y in zip(pa[f], pb[f]) if abs(x['box'][0] - y['box'][0]) < 1e-6 and abs(x['box'][1] - y['box'][1]) < 1e-6)
yaw_ch = sum(1 for f in pa for x, y in zip(pa[f], pb[f]) if abs(x['box'][6] - y['box'][6]) > 1e-6)
print('pedestrian boxes with same centre', same_xy, '| with changed heading', yaw_ch)
EOF
echo V16_ALL_DONE
