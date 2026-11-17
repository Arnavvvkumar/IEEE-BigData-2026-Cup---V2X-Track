#!/bin/bash
# Verify the v4 zip and compare its predictions with v3 (box counts by class at score thresholds).
here="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final"
P="$HOME/projects/detector-program/env310/bin/python"
"$P" "$here/verify_zip.py" "/mnt/c/Users/arnav/Downloads/bdc2026_trackB_submission_v4.zip"
echo "verify exit $?"
"$P" - "$here" << 'PY'
import json, sys
from pathlib import Path
for v in ('candidate_v3', 'candidate_v4'):
    p = json.loads((Path(sys.argv[1]) / v / 'predictions.json').read_text())
    row = {}
    for c in ('vehicle', 'pedestrian', 'truck'):
        s = [d['score'] for f in p for d in p[f] if d['class'] == c]
        row[c] = [len(s)] + [round(sum(x >= t for x in s) / len(p), 2) for t in (0.1, 0.3, 0.5)] + [round(max(s), 3)]
    print(v, 'n, per-frame >=0.1/0.3/0.5, max', row)
PY
