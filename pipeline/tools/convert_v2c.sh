#!/bin/bash
set -euo pipefail
H="${BDC_ROOT}/experiments/claude_carla_overnight"
E=${BDC_PYTHON}
CONV=${BDC_DATA}/claude_carla64_town02_v2c
cd "$H"
nice -n 10 taskset -c 4-13 $E convert_carla.py --native native_v1 --out $CONV --containers-seed 4242
nice -n 10 taskset -c 4-13 $E audit_carla.py --native native_v1 --converted $CONV --out audit_carla_v2c.json
$E - <<'EOF'
import json, numpy as np, pathlib
r = json.loads(pathlib.Path.home().joinpath('projects/bdc2026-track-b/data/claude_carla64_town02_v2c/frames.json').read_text())
clips = {x['layout_id']: len(x['synthetic_containers']) for x in r}
ann = [d for x in r for d in x['object_details'] if d['blueprint'] == 'synthetic.container']
print('containers per clip', sorted(clips.values()), 'total placed', sum(clips.values()))
print('container boxes', len(ann), 'annotated', sum(d['annotated'] for d in ann), 'median hits', np.median([d['hits'] for d in ann]),
      'median range', round(float(np.median([d['range_m'] for d in ann])), 1))
c = {k: sum(g['class'] == k for x in r for g in x['ground_truth']) for k in ('vehicle', 'pedestrian', 'truck')}
print('boxes', c)
EOF
