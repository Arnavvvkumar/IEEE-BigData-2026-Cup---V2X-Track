#!/bin/bash
# Candidate v9 (probe) = v8 (v7 + PV-RCNN epoch 2 for pedestrian/vehicle) + truck boost/hedge (calib.truck_boost_hedge).
# candidate_v7 / v8 and their zips are NOT touched: the build writes to candidate_v9. Declaration tag set for v9.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/final"
dl="${BDC_OUT}"
sed -i 's/^Submission tag:.*/Submission tag:         v9-pvrcnn2-realism-2026-11-13/' "$dl/declaration.txt"
EXTRA="${EXTRA:-pvrcnn_st_osc_ep2:pedestrian,vehicle}" TRUCKPP=1 bash "$here/build_candidate_v7.sh" candidate_v9 bdc2026_trackB_submission_v9.zip 1
sed -i 's/^Submission tag:.*/Submission tag:         v8-hedge-pvrcnn-2026-11-12/' "$dl/declaration.txt"   # restore v8's tag for the file's resting state
${BDC_PYTHON} - "$dl" << 'PY'
import json, sys, zipfile
for v in ('v7', 'v8', 'v9'):
    z = zipfile.ZipFile(f'{sys.argv[1]}/bdc2026_trackB_submission_{v}.zip'); p = json.loads(z.read('predictions.json'))
    tag = [l.split(':', 1)[1].strip() for l in z.read('declaration.txt').decode().splitlines() if l.lower().startswith('submission tag')]
    print(v, 'boxes', sum(len(x) for x in p.values()), {c: sum(d['class'] == c for x in p.values() for d in x) for c in ('vehicle', 'pedestrian', 'truck')}, 'tag', tag)
PY
