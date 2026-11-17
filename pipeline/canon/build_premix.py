"""Assemble a pre-canonical training mix (symlinked clouds + labels) from already-relabelled frames and new CARLA v3 frames.
v3 box sizes are mapped to the V2X-Real annotation convention with the same rule as boxsize/relabel_mix.py (quantile map per
class and dimension, fitted on the v3 boxes against the SELECTION-half real box sizes; clamp 0.8-2.0x; bottom kept).
Val = the 500 Generator V2 validation frames (symlinks). Then run canon/build_canon.py on the result.
usage: build_premix.py --relabelled <relabel mix dir> --take v2|carla|both --v3 <v3 dir> [--v3-frames N] --out <dir>"""
import argparse
import json
from pathlib import Path

import numpy as np

MAP_DIMS = {'vehicle': (0, 1, 2), 'pedestrian': (0, 1), 'truck': (1,)}
GRID = np.linspace(0, 100, 101)
W = Path(__file__).resolve().parents[1]


def read_labels(p):
    out = []
    for ln in Path(p).read_text().splitlines():
        *n, c = ln.split()
        out.append(([float(x) for x in n], c))
    return out


ap = argparse.ArgumentParser()
ap.add_argument('--relabelled', type=Path, required=True)
ap.add_argument('--take', default='both')
ap.add_argument('--v3', type=Path)
ap.add_argument('--v3-frames', type=int, default=0)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    raise FileExistsError(a.out)
for d in ('points', 'labels', 'ImageSets'):
    (a.out / d).mkdir(parents=True)
train = []
for f in (a.relabelled / 'ImageSets/train.txt').read_text().split():
    is_carla = f.startswith('carla')
    if a.take == 'both' or (a.take == 'carla') == is_carla:
        (a.out / 'points' / f'{f}.npy').symlink_to((a.relabelled / 'points' / f'{f}.npy').resolve())
        (a.out / 'labels' / f'{f}.txt').symlink_to((a.relabelled / 'labels' / f'{f}.txt').resolve())
        train.append(f)
report = {'relabelled_frames': len(train)}
if a.v3:
    real = json.loads((W / 'boxsize/real_box_stats_selection.json').read_text())
    assert 'selection half' in real['source']
    ids = (a.v3 / 'ImageSets/train.txt').read_text().split()
    if a.v3_frames and a.v3_frames < len(ids):  # whole clips, evenly spread
        clips = sorted({f.rsplit('_F', 1)[0] for f in ids})
        keep = set(clips[i] for i in np.linspace(0, len(clips) - 1, a.v3_frames // 40).astype(int))
        ids = [f for f in ids if f.rsplit('_F', 1)[0] in keep]
    labels = {f: read_labels(a.v3 / 'labels' / f'{f}.txt') for f in ids}
    maps = {}
    for c, dims in MAP_DIMS.items():
        syn = np.array([b[3:6] for L in labels.values() for b, cc in L if cc == c])
        rl = np.asarray(real['classes'][c]['raw_lwh'])
        maps[c] = {d: (np.percentile(syn[:, d], GRID), np.percentile(rl[:, d], GRID)) for d in dims}
        report[c] = {'v3_boxes': len(syn), 'v3_median_lwh': np.median(syn, 0).round(3).tolist(), 'real_median_lwh': np.median(rl, 0).round(3).tolist()}
    for f in ids:
        (a.out / 'points' / f'{f}.npy').symlink_to((a.v3 / 'points' / f'{f}.npy').resolve())
        lines = []
        for b, c in labels[f]:
            old = list(b)
            for d, (qs, qr) in maps[c].items():
                b[3 + d] = float(np.clip(np.interp(old[3 + d], qs, qr), 0.8 * old[3 + d], 2.0 * old[3 + d]))
            b[2] = old[2] - old[5] / 2 + b[5] / 2
            lines.append(' '.join(format(float(x), '.17g') for x in b) + ' ' + c + '\n')
        (a.out / 'labels' / f'{f}.txt').write_text(''.join(lines))
        train.append(f)
    report['v3_frames'] = len(ids)
val = (a.relabelled / 'ImageSets/val.txt').read_text().split()
for f in val:
    for d, e in (('points', 'npy'), ('labels', 'txt')):
        (a.out / d / f'{f}.{e}').symlink_to((a.relabelled / d / f'{f}.{e}').resolve())
(a.out / 'ImageSets/train.txt').write_text('\n'.join(train) + '\n')
(a.out / 'ImageSets/val.txt').write_text('\n'.join(val) + '\n')
report['train'] = len(train)
(a.out / 'premix_manifest.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report))
