import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
recs = json.loads((d / 'frames.json').read_text())
for r in recs[:10:3]:
    for o in r['object_details']:
        if o['class'] == 'pedestrian':
            b = o['box']
            print(r['frame'], o['id'], o['blueprint'], 'range', round(o['range_m'], 1), 'z', round(b[2], 2), 'h', round(b[5], 2),
                  'hits', o['hits'], 'sem', o['semantic_hits'], 'cir', o['corners_in_range'])
