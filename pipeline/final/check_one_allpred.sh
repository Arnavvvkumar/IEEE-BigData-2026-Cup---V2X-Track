#!/bin/bash
P="$HOME/projects/detector-program/env310/bin/python"
"$P" - << 'PY'
import json
from pathlib import Path
p = Path("/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop/dev350/allpred")
for f in sorted(p.glob('*.json')):
    d = json.loads(f.read_text())
    k = list(d)
    print(f.name, "keys", len(k), "frames", len(d[k[0]]), "bytes", f.stat().st_size)
PY
