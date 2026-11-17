import json
import sys
from collections import Counter
from pathlib import Path

d = Path(sys.argv[1])
audit = json.loads(Path(sys.argv[2]).read_text())
print('per-blueprint inside (n, median):')
for k, v in audit['per_blueprint_inside_median'].items():
    if v[1] < 0.98 or 'walker' in k:
        print('  ', k, v)
recs = json.loads((d / 'frames.json').read_text())
c = Counter()
for r in recs:
    for o in r['object_details']:
        if o['class'] == 'pedestrian' and o['annotated']:
            c[(r['layout_id'], o['id'], o['blueprint'])] += 1
print('annotated pedestrian tracks:', c.most_common(10))
sem = Counter()
for r in recs:
    for o in r['object_details']:
        if o['class'] == 'pedestrian' and o['semantic_hits'] > 0:
            sem[(r['layout_id'], o['blueprint'])] += 1
print('pedestrian with semantic hits (clip, bp):', sem.most_common(10))
