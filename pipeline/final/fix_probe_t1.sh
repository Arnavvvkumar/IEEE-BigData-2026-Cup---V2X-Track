#!/bin/bash
# The T1 probe kept TRUCKBOOST=1.0 because the build line sets it inline (overriding the export). Rebuild with the inline value replaced.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/final"
g="$here/_build_10t1_gen.sh"
sed -i 's/TRUCKBOOST=1\.0/TRUCKBOOST=2.0/' "$g"
grep -o 'TRUCKBOOST=[0-9.]*' "$g" | sort | uniq -c
bash "$g" 2>&1 | grep -E 'rule|zip |scorer|PROBE_BUILT|Traceback|Error' | cut -c1-200
SYN=final_synthetic_v4 ${BDC_PYTHON} "$here/verify_zip_any.py" bdc2026_trackB_submission_v10t1.zip bdc2026_trackB_submission_v10r.zip | grep -E 'detections|truck|declaration|identical'
${BDC_PYTHON} - << 'PY'
import json, zipfile
D = __import__('os').environ['BDC_OUT'] + '/'
a = json.loads(zipfile.ZipFile(D + 'bdc2026_trackB_submission_v10r.zip').read('predictions.json'))
b = json.loads(zipfile.ZipFile(D + 'bdc2026_trackB_submission_v10t1.zip').read('predictions.json'))
ch = sum(1 for f in a for x, y in zip(sorted((d['score'] for d in a[f] if d['class'] == 'truck'), reverse=True),
                                       sorted((d['score'] for d in b[f] if d['class'] == 'truck'), reverse=True)) if abs(x - y) > 1e-9)
print('truck scores that differ between v10r and v10t1:', ch)
PY
