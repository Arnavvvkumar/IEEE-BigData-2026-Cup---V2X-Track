#!/bin/bash
F="${BDC_ROOT}/experiments/claude_carla_overnight/final"
cd ${BDC_OUT}
sed -i 's/\r$//' "$F/build_v13.sh"
ADD='pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian' bash "$F/build_v13.sh" 13
# confirm predictions identical to v12 and synthetic identical to v9
python3 - <<'EOF'
import zipfile
a=zipfile.ZipFile('bdc2026_trackB_submission_v12.zip'); b=zipfile.ZipFile('bdc2026_trackB_submission_v13.zip'); c=zipfile.ZipFile('bdc2026_trackB_submission_v9.zip')
print('predictions identical to v12:', a.read('predictions.json')==b.read('predictions.json'))
syn=[n for n in b.namelist() if n.startswith('synthetic/') and not n.endswith('/')]
print('synthetic files', len(syn), 'identical to v9:', all(b.read(n)==c.read(n) for n in syn))
print([l for l in b.read('declaration.txt').decode().splitlines() if l.startswith(('Date','Submission'))])
EOF
bash "$F/kit_validate.sh" bdc2026_trackB_submission_v13.zip 2>&1 | grep -E 'valid_submission|score_realism'
