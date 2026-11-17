"""Compare folder-median vs per-scan levelled test predictions of one voter: detection counts and matched-box shifts."""
import json, sys
from pathlib import Path
import numpy as np
h = Path(__file__).parent
name = sys.argv[1]
a = json.loads((h / 'test_allpred' / name).read_text()); b = json.loads((h / 'test_allpred_ps' / name).read_text())
for g in list(a)[:6]:
    na = sum(len(v) for v in a[g].values()); nb = sum(len(v) for v in b[g].values())
    d = []
    for f in a[g]:
        A = [x for x in a[g][f] if x['score'] > 0.3]; B = [x for x in b[g][f] if x['score'] > 0.3]
        if not A or not B:
            continue
        cb = np.array([x['box'][:3] for x in B])
        for x in A:
            dist = np.linalg.norm(cb - np.array(x['box'][:3]), axis=1); d.append(dist.min())
    d = np.array(d)
    print(f'ground {g}: dets {na} -> {nb} | score>0.3 centre shift median {np.median(d):.3f} m, 90% {np.percentile(d, 90):.3f} m, >0.5 m {np.mean(d > 0.5):.3f}')
