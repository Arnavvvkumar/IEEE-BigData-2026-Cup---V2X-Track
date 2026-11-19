#!/bin/bash
P="${BDC_PYTHON}"
"$P" - << 'PY'
import json
from pathlib import Path
p = Path(__import__('os').environ['BDC_ROOT'] + "/experiments/claude_carla_overnight/coop/dev350/allpred")
for f in sorted(p.glob('*.json')):
    d = json.loads(f.read_text())
    k = list(d)
    print(f.name, "keys", len(k), "frames", len(d[k[0]]), "bytes", f.stat().st_size)
PY
