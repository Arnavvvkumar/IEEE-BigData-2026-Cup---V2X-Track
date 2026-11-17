#!/bin/bash
# v14 = v13 + P1 TTA x3 + P3 TTA x3 (+ EXTRA voters passed in env). v3 realism scans (same as v9/v13).
F="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
name=${1:-14}
cd /mnt/c/Users/arnav/Downloads
sed -i 's/\r$//' "$F/build_v13.sh"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
ls -la "$F/test_allpred_ps1/" | grep -E 'P1_tta|P3_tta|P4|P5|P6'
ADD="$(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) ${EXTRA:-}" bash "$F/build_v13.sh" $name
python3 - "$name" <<'EOF'
import zipfile, json, sys, collections
n=sys.argv[1]
a=zipfile.ZipFile('bdc2026_trackB_submission_v13.zip'); b=zipfile.ZipFile(f'bdc2026_trackB_submission_v{n}.zip'); c=zipfile.ZipFile('bdc2026_trackB_submission_v9.zip')
syn=[x for x in b.namelist() if x.startswith('synthetic/') and not x.endswith('/')]
print('synthetic files', len(syn), 'identical to v9:', all(b.read(x)==c.read(x) for x in syn))
pa=json.loads(a.read('predictions.json')); pb=json.loads(b.read('predictions.json'))
def cnt(p):
    d=p if isinstance(p,dict) else None
    c=collections.Counter()
    items = d.values() if d is not None else [p]
    for v in items:
        for x in (v if isinstance(v,list) else v.get('detections', v.get('objects', []))):
            c[x.get('class', x.get('label'))]+=1
    return c
print('v13 counts', dict(cnt(pa))); print(f'v{n} counts', dict(cnt(pb)))
print('predictions identical to v13:', a.read('predictions.json')==b.read('predictions.json'))
EOF
bash "$F/kit_validate.sh" bdc2026_trackB_submission_v$name.zip 2>&1 | grep -E 'valid_submission|score_realism'
