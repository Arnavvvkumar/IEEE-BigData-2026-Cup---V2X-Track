"""One unmodified-kit vs chunked MMD equivalence check on allowed dev scans."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from replay import EXP, KIT, read, screened

ref = screened()
sys.path.insert(0,str(EXP / 'twin'))
import range_bg
import metrics
if len(sys.argv) > 1:
    metrics.N_SUBSAMPLE = int(sys.argv[1])

cfg = read(KIT / 'reference_data_dummy/config.json')
pts = [np.fromfile(p,dtype='<f4').reshape(-1,3)
       for p in sorted((EXP / 'twin/final_synthetic_v4/synthetic').glob('*.bin'))]
assert len(pts) == 50
ids = range_bg.split['confirmation_ids']
assert set(ids) <= set(read(ref / 'source_index.json'))
dev = [range_bg.clouds[k] for k in ids]
start = time.perf_counter()
raw = metrics.compute_realism(pts,dev,tuple(cfg['point_cloud_range']),seed=1234)
norm = {k: float(np.clip((v[1]-raw[k])/(v[1]-v[0]),0,1))
        for k,v in cfg['normalization'].items() if k in raw}
value = sum(norm.values())/len(norm)
lean = range_bg.lean(pts,dev,seed=1234)
assert round(value,4) == lean['realism']
assert all(round(x,3)==lean[k+'_n'] for k,x in norm.items())
print(json.dumps({'n_synthetic':len(pts),'n_reference':len(dev),'sample_size':metrics.N_SUBSAMPLE,'raw_kit':raw,'normalized':norm,
                  'kit':value,'lean':lean,'seconds':round(time.perf_counter()-start,1)},indent=2))
