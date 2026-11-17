"""Derive per-blueprint tight boxes for RIGID vehicles from CARLA semantic-LiDAR instance returns.

CARLA's actor bounding boxes are loose or offset for some large blueprints (e.g. the bus box is ~0.9 m too wide and the
European HGV box is ~0.35 m short of its roof). For each vehicle blueprint we pool all own-instance semantic returns in the
CARLA box frame over every saved semantic frame (different headings/ranges reveal the whole body), take the 0.1/99.9
percentile extents per axis, and store the local centre offset and tight dimensions. Walkers are articulated and are left
unchanged. The bottom face is kept at the CARLA box bottom (the underside is never observed by a roadside sensor).
Output: bbox_refinement.json, consumed by convert_carla.py (pinned by sha256 in its provenance).
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import convert_carla as C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--native', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-points', type=int, default=2000)
    a = ap.parse_args(argv)
    pts, raw_dims = defaultdict(list), {}
    for nat in a.native:
        for cdir in sorted(nat.glob('clip_*')):
            for sp in sorted((cdir / 'semantic').glob('*.npy')):
                rec = json.loads((cdir / 'actors' / sp.name.replace('.npy', '.json')).read_text())
                sem = np.load(sp)
                for act in rec['actors']:
                    if not act['blueprint'].startswith('vehicle.'):
                        continue
                    m = sem['obj'] == act['id']
                    if m.sum() < 20:
                        continue
                    box = C.box_from_record(act, rec, refine=False)
                    d = np.c_[sem['x'][m], -sem['y'][m], sem['z'][m]].astype(float) - box[:3]
                    c, s = math.cos(-box[6]), math.sin(-box[6])
                    pts[act['blueprint']].append(np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]])
                    raw_dims[act['blueprint']] = box[3:6].tolist()
    table = {}
    for bp, chunks in sorted(pts.items()):
        p = np.concatenate(chunks)
        if len(p) < a.min_points:
            continue
        lo = np.percentile(p, 0.1, axis=0)
        hi = np.percentile(p, 99.9, axis=0)
        h0 = raw_dims[bp][2]
        lo[2] = -h0 / 2  # keep the CARLA bottom face
        dims = hi - lo
        centre = (hi + lo) / 2
        old = np.asarray(raw_dims[bp])
        inside_old = float(np.mean(np.all(np.abs(p) <= old / 2 + 0.15, axis=1)))
        inside_new = float(np.mean(np.all(np.abs(p - centre) <= dims / 2 + 0.15, axis=1)))
        # Conservative: fix only boxes that fail to contain their own returns, or that are >0.5 m too large on a
        # well-observed (>=50 frames) blueprint. A roadside view sees mostly one side, so smaller changes are not trusted.
        changed = bool(inside_old < 0.98 or (len(chunks) >= 50 and np.any(old - dims > 0.5)))
        table[bp] = {'points': int(len(p)), 'frames': len(chunks), 'carla_lwh': old.round(3).tolist(),
                     'tight_lwh': dims.round(3).tolist(), 'local_centre_offset': centre.round(3).tolist(),
                     'inside_0p15_carla': round(inside_old, 4), 'inside_0p15_tight': round(inside_new, 4), 'apply': changed}
    out = {'rule': 'apply when CARLA box holds <98% of own returns (0.15 m margin) or, with >=50 frames, a CARLA dim exceeds '
                   'the 0.1/99.9-percentile extent by >0.5 m; vehicles only; bottom face kept',
           'sources': [str(n) for n in a.native], 'blueprints': table}
    a.out.write_text(json.dumps(out, indent=1))
    for bp, t in table.items():
        if t['apply']:
            print(bp, t)


if __name__ == '__main__':
    main()
