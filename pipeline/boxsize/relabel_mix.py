"""Build a 5k Generator V2 + 1k CARLA (with synthetic containers) training mix whose TRAIN box sizes follow the V2X-Real
annotation convention. The fixed 500 Generator V2 validation files are linked verbatim (frozen, unchanged).

Size mapping (per class, per dimension) = monotone quantile mapping from the pooled synthetic training boxes to the real
box sizes of the SELECTION half only (real_box_stats_selection.json; permitted box-statistics use of non-forbidden public
training frames; the confirmation half is never read). Mapped dimensions:
  vehicle: l, w, h        pedestrian: l, w (height already matches)        truck: w only (lengths/heights are geometry;
  real trucks mix buses, box trucks and containers)        synthetic containers: unchanged (their size is their geometry)
Centres x, y and yaw are unchanged; z moves so the box bottom stays put. Each mapped value is clamped to [0.8, 2.0] x original.
"""
import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

MAP_DIMS = {'vehicle': (0, 1, 2), 'pedestrian': (0, 1), 'truck': (1,)}
GRID = np.linspace(0, 100, 101)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_labels(p):
    out = []
    for ln in Path(p).read_text().splitlines():
        *n, c = ln.split()
        out.append(([float(x) for x in n], c))
    return out


def fmt(box, c):
    return ' '.join(format(float(x), '.17g') for x in box) + ' ' + c + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--carla', type=Path, required=True)
    ap.add_argument('--real-stats', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    base, carla, out = a.base.resolve(), a.carla.resolve(), a.out
    if out.exists():
        raise FileExistsError(out)
    real = json.loads(a.real_stats.read_text())
    if 'selection half' not in real['source']:
        raise ValueError('real statistics must come from the selection half only')
    btrain = (base / 'ImageSets/train.txt').read_text().split()
    bval = (base / 'ImageSets/val.txt').read_text().split()
    cids = (carla / 'ImageSets/train.txt').read_text().split()
    assert len(btrain) == 5000 and len(bval) == 500 and len(cids) == 1000 and not set(cids) & set(btrain + bval)
    # container flags per CARLA label line (gt order == annotated object_details order)
    cont = {}
    for r in json.loads((carla / 'frames.json').read_text()):
        cont[r['frame']] = [d['blueprint'] == 'synthetic.container' for d in r['object_details'] if d['annotated']]
    labels = {f: read_labels(base / 'labels' / f'{f}.txt') for f in btrain}
    labels.update({f: read_labels(carla / 'labels' / f'{f}.txt') for f in cids})
    for f in cids:
        assert len(cont[f]) == len(labels[f])
    # fit quantile maps
    maps, report = {}, {}
    for c, dims in MAP_DIMS.items():
        syn = np.array([b[3:6] for f, L in labels.items() for i, (b, cc) in enumerate(L)
                        if cc == c and not (f in cont and cont[f][i])])
        rl = np.asarray(real['classes'][c]['raw_lwh'])
        maps[c] = {d: (np.percentile(syn[:, d], GRID), np.percentile(rl[:, d], GRID)) for d in dims}
        report[c] = {'synthetic_n': len(syn), 'real_n': len(rl), 'dims': list(dims),
                     'synthetic_median_lwh': np.median(syn, 0).round(3).tolist(), 'real_median_lwh': np.median(rl, 0).round(3).tolist()}
    for d in ('points', 'labels', 'ImageSets'):
        (out / d).mkdir(parents=True)
    changed = {c: [] for c in MAP_DIMS}
    for f in btrain + cids:
        src = base if f in labels and f in btrain else carla
        (out / 'points' / f'{f}.npy').symlink_to(src / 'points' / f'{f}.npy')
        lines = []
        for i, (b, c) in enumerate(labels[f]):
            b = list(b)
            if c in maps and not (f in cont and cont[f][i]):
                old = list(b)
                for d, (qs, qr) in maps[c].items():
                    v = float(np.interp(b[3 + d], qs, qr))
                    b[3 + d] = float(np.clip(v, 0.8 * old[3 + d], 2.0 * old[3 + d]))
                b[2] = old[2] - old[5] / 2 + b[5] / 2
                changed[c].append(np.array(b[3:6]) / np.array(old[3:6]))
            lines.append(fmt(b, c))
        (out / 'labels' / f'{f}.txt').write_text(''.join(lines))
    for f in bval:
        for d, e in (('points', 'npy'), ('labels', 'txt')):
            (out / d / f'{f}.{e}').symlink_to(base / d / f'{f}.{e}')
    train = btrain + cids
    (out / 'ImageSets/train.txt').write_text('\n'.join(train) + '\n')
    shutil.copyfile(base / 'ImageSets/val.txt', out / 'ImageSets/val.txt')
    infos = []
    for f in train:
        L = read_labels(out / 'labels' / f'{f}.txt')
        infos.append({'point_cloud': {'num_features': 4, 'lidar_idx': f},
                      'annos': {'name': np.array([c for _, c in L]), 'gt_boxes_lidar': np.array([b for b, _ in L], dtype=np.float32).reshape(-1, 7)}})
    (out / 'custom_infos_train.pkl').write_bytes(pickle.dumps(infos, protocol=4))
    (out / 'custom_infos_val.pkl').symlink_to(base / 'custom_infos_val.pkl')
    for c, v in changed.items():
        v = np.asarray(v)
        report[c]['median_ratio_new_over_old_lwh'] = np.median(v, 0).round(3).tolist() if len(v) else None
        report[c]['relabelled_boxes'] = len(v)
    man = {'scope': 'TRAIN sizes mapped to the V2X-Real annotation convention; val frozen', 'mapping': report,
           'real_stats_sha256': sha(a.real_stats), 'base_path': str(base), 'carla_path': str(carla),
           'carla_frames_json_sha256': sha(carla / 'frames.json'), 'builder_sha256': sha(__file__),
           'train_list_sha256': sha(out / 'ImageSets/train.txt'), 'val_list_sha256': sha(out / 'ImageSets/val.txt'),
           'train_infos_sha256': sha(out / 'custom_infos_train.pkl'), 'base_val_infos_sha256': sha(base / 'custom_infos_val.pkl'),
           'counts': {'train': len(train), 'generator_v2': len(btrain), 'carla': len(cids), 'val': len(bval)},
           'train_boxes': {c: int(sum(n == c for i in infos for n in i['annos']['name'].tolist())) for c in ('vehicle', 'pedestrian', 'truck')}}
    (out / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ('mapping', 'counts', 'train_boxes', 'train_infos_sha256')}, indent=1))


if __name__ == '__main__':
    main()
