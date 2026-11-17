"""Measure native truck proposals per exported checkpoint before ensemble fusion."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'codex_postmortem'))
from truck_replay import ap, frame_rows, fusion, iou, read, screened


def main(size):
    screened()
    dev = BASE / 'coop' / f'dev{size}'
    labels = {f: [g for g in gs if g['points'] >= 5] for f, gs in read(dev / 'gt_coop.json').items()}
    ids = sorted(labels)
    line = next(line for line in reversed((dev / 'eval_summary.txt').read_text().splitlines())
                if line.startswith('v12p3x3_norule | '))
    voters = [tag for spec in line.split(' | ')[1].removesuffix(' norule').split()
              for tag, classes in [spec.split(':')] if 'truck' in classes.split(',')]
    # Evaluate each checkpoint with its own height passes and no copies or extra hedges.
    fusion.C, fusion.TRUCK_SCORE, fusion.ONE_VOTE = ['truck'], '', set()
    for tag in voters + ['second_truck_T2_ep12', 'pvrcnn_truck_T1_ep2']:
        assert not tag.startswith('second_twin_ll_')
        raw = read(dev / 'allpred' / f'{tag}_allpred.json')
        pred = fusion.fuse(list(raw.values()), ids)
        pools = defaultdict(list)
        for f in ids:
            gt = [g for g in labels[f] if g['class'] == 'truck']
            # Distinguish proposal recall before clustering from recall after clustering.
            before = [d['box'] for p in raw.values() for d in p[f] if d['class'] == 'truck']
            after = [d['box'] for d in pred[f]]
            hi = iou(before, [g['box'] for g in gt]).max(0, initial=0)
            hj = iou(after, [g['box'] for g in gt]).max(0, initial=0)
            for label, a, b in zip(gt, hi, hj):
                pools[label['type']].append((a, b))
        print(json.dumps({'set': size, 'voter': tag, 'passes': len(raw),
                          'native_truck_AP': ap(frame_rows(pred, labels, 'truck')),
                          'native_genuine_AP': ap(frame_rows(pred, labels, 'truck', True)),
                          'subtypes_n_raw_IoU50_fused_IoU50': {
                              key: [len(value), *map(int, (np.asarray(value) >= .5).sum(0))]
                              for key, value in pools.items()}}), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
