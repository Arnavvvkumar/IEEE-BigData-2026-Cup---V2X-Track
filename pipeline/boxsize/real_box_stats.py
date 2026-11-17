"""Real V2X-Real box-size statistics (permitted 'box statistics' use of non-forbidden public TRAINING frames).

Uses ONLY the selection half of real_split_proposal_v1.json, so the confirmation half stays untouched by any label
convention we adopt. Boxes in the Track B range, sizes l/w/h per superclass and per publisher subtype, plus quantiles.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'V2X-Real'))
from opencood.data_utils import SUPER_CLASS_MAP  # noqa: E402
from opencood.utils.transformation_utils import x1_to_x2  # noqa: E402

KIT = ROOT / 'experiments/real_reference_audit/real_dev_cw1'
SPLIT = ROOT / 'experiments/claude_carla_overnight/realval/real_split_proposal_v1.json'
Q = [5, 10, 25, 50, 75, 90, 95]


def main():
    split = json.loads(SPLIT.read_text())
    idx = json.loads((KIT / 'source_index.json').read_text())
    cls = {s: g for g, names in SUPER_CLASS_MAP.items() for s in names}
    by_cls, by_sub = defaultdict(list), defaultdict(list)
    for kid in split['selection_ids']:
        lab = yaml.safe_load(Path(idx[kid]['label']).read_text())
        for o in lab['vehicles'].values():
            c = cls.get(o['obj_type'])
            if c is None:
                continue
            t = x1_to_x2([*(np.asarray(o['location']) + o['center']), *o['angle']], lab['lidar_pose'])
            ctr = t[:3, 3]
            if (np.abs(ctr[:2]) > 40).any() or ctr[2] < -8 or ctr[2] > 2:
                continue
            s = (2 * np.asarray(o['extent'])).tolist()
            by_cls[c].append(s)
            by_sub[o['obj_type']].append(s)
    out = {'source': 'selection half of real_split_proposal_v1.json (25 frames); confirmation half NOT used',
           'classes': {}, 'subtypes': {}}
    for c, v in by_cls.items():
        a = np.asarray(v)
        out['classes'][c] = {'n': len(a), 'quantiles_lwh': {str(q): np.percentile(a, q, axis=0).round(3).tolist() for q in Q},
                             'raw_lwh': a.round(4).tolist()}
    for s, v in by_sub.items():
        a = np.asarray(v)
        out['subtypes'][s] = {'n': len(a), 'median_lwh': np.median(a, 0).round(3).tolist(),
                              'p10_lwh': np.percentile(a, 10, 0).round(3).tolist(), 'p90_lwh': np.percentile(a, 90, 0).round(3).tolist()}
    p = Path(__file__).with_name('real_box_stats_selection.json')
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({'classes': {c: [d['n'], d['quantiles_lwh']['50']] for c, d in out['classes'].items()},
                      'subtypes': {s: [d['n'], d['median_lwh'], d['p10_lwh'], d['p90_lwh']] for s, d in out['subtypes'].items()}}, indent=1))


if __name__ == '__main__':
    main()
