"""Independent audit of the relabelled mix: frozen val, unchanged clouds, only sizes (and bottom-preserving z) changed."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

D = Path.home() / 'projects/bdc2026-track-b/data'
MIX, BASE, CARLA = D / 'claude_v2carla_relabel_mix6000_v1', D / 'codex_openpcdet_custom_v1', D / 'claude_carla64_town02_v2c'
CANON = D / 'claude_v2_scale5500_fix1/frames.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lab(p):
    return [(np.array([float(x) for x in ln.split()[:-1]]), ln.split()[-1]) for ln in Path(p).read_text().splitlines()]


chk, bad = {}, []
bt, bv = (BASE / 'ImageSets/train.txt').read_text().split(), (BASE / 'ImageSets/val.txt').read_text().split()
ct = (CARLA / 'ImageSets/train.txt').read_text().split()
tr, va = (MIX / 'ImageSets/train.txt').read_text().split(), (MIX / 'ImageSets/val.txt').read_text().split()
chk['train_ids_5000_base_plus_1000_carla'] = tr == bt + ct and len(set(tr)) == 6000
chk['val_list_identical'] = va == bv and sha(MIX / 'ImageSets/val.txt') == sha(BASE / 'ImageSets/val.txt')
chk['val_infos_identical'] = sha(MIX / 'custom_infos_val.pkl') == sha(BASE / 'custom_infos_val.pkl')
chk['val_files_identical'] = all(sha(MIX / d / f'{f}.{e}') == sha(BASE / d / f'{f}.{e}') for f in bv for d, e in (('points', 'npy'), ('labels', 'txt')))
chk['train_clouds_identical'] = all(sha(MIX / 'points' / f'{f}.npy') == sha((BASE if f in set(bt) else CARLA) / 'points' / f'{f}.npy') for f in tr)
ratios = {}
for f in tr:
    src = lab((BASE if f.startswith('v2') else CARLA) / 'labels' / f'{f}.txt')
    new = lab(MIX / 'labels' / f'{f}.txt')
    if len(src) != len(new) or [c for _, c in src] != [c for _, c in new]:
        bad.append(('count/class', f)); continue
    for (a, c), (b, _) in zip(src, new):
        if not (np.allclose(a[[0, 1, 6]], b[[0, 1, 6]], atol=1e-9) and abs((a[2] - a[5] / 2) - (b[2] - b[5] / 2)) < 1e-6
                and np.all(b[3:6] >= 0.8 * a[3:6] - 1e-9) and np.all(b[3:6] <= 2.0 * a[3:6] + 1e-9)):
            bad.append(('box', f)); break
        ratios.setdefault(c, []).append(b[3:6] / a[3:6])
chk['only_sizes_changed_bottom_kept'] = not bad
infos = pickle.loads((MIX / 'custom_infos_train.pkl').read_bytes())
chk['infos_match_labels'] = [i['point_cloud']['lidar_idx'] for i in infos] == tr and all(
    i['annos']['name'].tolist() == [c for _, c in lab(MIX / 'labels' / f"{i['point_cloud']['lidar_idx']}.txt")] and
    np.array_equal(i['annos']['gt_boxes_lidar'], np.array([b for b, _ in lab(MIX / 'labels' / f"{i['point_cloud']['lidar_idx']}.txt")], dtype=np.float32).reshape(-1, 7))
    for i in infos)
chk['canonical_frames_json_unchanged'] = sha(CANON) == '13abb7b25a0da8651221734bc1adf431e630c6006bc1027d215cfb5090778a45'
out = {'checks': chk, 'all_passed': all(chk.values()), 'bad': bad[:10],
       'median_size_ratio_lwh': {c: np.median(v, 0).round(3).tolist() for c, v in ratios.items()},
       'manifest_sha256': sha(MIX / 'manifest.json'), 'train_infos_sha256': sha(MIX / 'custom_infos_train.pkl')}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
