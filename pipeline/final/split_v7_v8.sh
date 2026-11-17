#!/bin/bash
# The PV-RCNN rebuild overwrote candidate_v7 / the v7 zip after v7 was submitted. Keep both, clearly named:
#   candidate_v8 + bdc2026_trackB_submission_v8.zip = v7 + PV-RCNN epoch 2 (pedestrian, vehicle)   [new, not submitted]
#   candidate_v7 + bdc2026_trackB_submission_v7.zip = exactly what was submitted at 23:33 (68,831 boxes)
set -euo pipefail
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
dl="/mnt/c/Users/arnav/Downloads"
P="$HOME/projects/detector-program/env310/bin/python"
cd "$here"
rm -rf candidate_v8 candidate_v8_meta
cp -r candidate_v7 candidate_v8; cp -r candidate_v7_meta candidate_v8_meta
cp "$dl/bdc2026_trackB_submission_v7.zip" "$dl/bdc2026_trackB_submission_v8.zip"
cp candidate_v7_meta/predictions_without_pvrcnn.json candidate_v7/predictions.json
bash finish_zip.sh candidate_v7 bdc2026_trackB_submission_v7.zip 2>/dev/null || true
for v in v7 v8; do
  "$P" - "$dl/bdc2026_trackB_submission_$v.zip" << 'PY'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); p = json.loads(z.read('predictions.json'))
print(sys.argv[1].split('/')[-1], 'boxes', sum(len(v) for v in p.values()), {c: sum(d['class'] == c for v in p.values() for d in v) for c in ('vehicle', 'pedestrian', 'truck')})
PY
done
