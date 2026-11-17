#!/bin/bash
# Build the v8 zip (v7 + PV-RCNN epoch 2) from candidate_v8 and print box counts for v7 and v8.
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
bash finish_zip.sh candidate_v8 bdc2026_trackB_submission_v8.zip 2>/dev/null
for v in v7 v8; do
  ~/projects/detector-program/env310/bin/python - "/mnt/c/Users/arnav/Downloads/bdc2026_trackB_submission_$v.zip" << 'PY'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); p = json.loads(z.read('predictions.json'))
print(sys.argv[1].split('/')[-1], 'boxes', sum(len(v) for v in p.values()), {c: sum(d['class'] == c for v in p.values() for d in v) for c in ('vehicle', 'pedestrian', 'truck')})
print('  declaration tag line:', [l for l in z.read('declaration.txt').decode().splitlines() if 'tag' in l.lower()])
PY
done
