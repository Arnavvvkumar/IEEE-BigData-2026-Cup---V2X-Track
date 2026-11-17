"""Label-free inspection of the fixed truck-sized objects in the 50 test frames: at each spot, how many points are there,
how large is the point cluster, and how do the truck vs vehicle predictions score there (final and per raw model pass)?
usage: spot_inspect.py"""
import json
import sys
from pathlib import Path

import numpy as np

W = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE')
F = W / 'experiments/claude_carla_overnight/final'
sys.path.insert(0, str(F))
import calib  # noqa: E402
import fuse_test as FT  # noqa: E402

SPOTS = {'A can (-4.4, 26.1)': (-4.4, 26.1), 'B (15.7,-24.6)': (15.7, -24.6), 'C (4.2,-24.2)': (4.2, -24.2), 'D (27.7,-23.6)': (27.7, -23.6), 'E (-30.6,-22.8)': (-30.6, -22.8)}
test = json.loads((F / 'candidate_v6/predictions.json').read_text())
clouds = FT.load(W / 'starting_kit/track_b/detection_test_frames')
MODELS = ['canon_all_s1234_ep12', 'second_twin_s1_ep12', 'canon_relabel_s1234_ep12', 'second_twin_s2_ep12', 'second_twin_st_s1_ep12', 'second_twin_pv_s1_ep14',
          'vnx010_all_s1234_ep10', 'vnx010_twin_s1_ep12', 'vnx010_pv_ft_ep5', 'vnx010_pv_ft2_ep5']
raw = {m: json.loads((F / 'test_allpred' / f'{m}_allpred.json').read_text()) for m in MODELS}
for label, (sx, sy) in SPOTS.items():
    near = lambda x, r=3.0: np.hypot(x['box'][0] - sx, x['box'][1] - sy) < r
    npts, ext, tr, ve, tb, vb = [], [], [], [], [], []
    for f in test:
        p = calib.nonground(np.asarray(clouds[f])[:, :3])
        q = p[np.hypot(p[:, 0] - sx, p[:, 1] - sy) < 5.0]
        npts.append(len(q))
        if len(q) > 5:
            c = q[:, :2] - q[:, :2].mean(0)
            w, v = np.linalg.eigh(np.cov(c.T))
            ext.append((np.ptp(c @ v[:, 1]), np.ptp(c @ v[:, 0]), q[:, 2].max() - q[:, 2].min()))
        t = [x for x in test[f] if x['class'] == 'truck' and near(x)]
        v_ = [x for x in test[f] if x['class'] == 'vehicle' and near(x)]
        tr.append(max((x['score'] for x in t), default=0)); ve.append(max((x['score'] for x in v_), default=0))
        if t:
            tb.append(max(t, key=lambda x: x['score'])['box'])
        if v_:
            vb.append(max(v_, key=lambda x: x['score'])['box'])
    e = np.median(np.array(ext), 0) if ext else [0, 0, 0]
    print(f'== {label}: non-ground points within 5 m median {int(np.median(npts))} | cluster extent {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} m')
    print(f'   final: truck score median {np.median(tr):.2f} (box LWH {np.median(np.array(tb), 0)[3:6].round(1).tolist() if tb else None}) | vehicle score median {np.median(ve):.2f} (box LWH {np.median(np.array(vb), 0)[3:6].round(1).tolist() if vb else None})')
    for m in MODELS:
        st, sv, n = [], [], 0
        for g, pf in raw[m].items():
            for f in pf:
                n += 1
                st.append(max((x['score'] for x in pf[f] if x['class'] == 'truck' and near(x)), default=0))
                sv.append(max((x['score'] for x in pf[f] if x['class'] == 'vehicle' and near(x)), default=0))
        st, sv = np.array(st), np.array(sv)
        print(f'     {m:26s} truck: found {100 * (st > 0.1).mean():3.0f}% mean {st.mean():.2f} | vehicle: found {100 * (sv > 0.1).mean():3.0f}% mean {sv.mean():.2f}')
