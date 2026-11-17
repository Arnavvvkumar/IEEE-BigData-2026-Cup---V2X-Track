import json
import sys
from pathlib import Path

import numpy as np

SEM = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('cos', '<f4'), ('obj', '<u4'), ('tag', '<u4')])
c = Path(sys.argv[1])
for f in sorted((c / 'semantic').glob('*.npy'))[:2]:
    s = np.load(f)
    tags, n = np.unique(s['tag'], return_counts=True)
    print(f.name, dict(zip(tags.tolist(), n.tolist())))
    rec = json.loads((c / 'actors' / f.name.replace('.npy', '.json')).read_text())
    ids = {a['id'] for a in rec['actors'] if a['blueprint'].startswith('walker')}
    ped = s[s['tag'] == 12]
    print(' ped-tag points', len(ped), 'obj ids', np.unique(ped['obj'])[:10], 'walker ids', sorted(ids))
