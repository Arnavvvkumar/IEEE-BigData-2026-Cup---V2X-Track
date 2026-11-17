"""Attribute truck AP to cached detector, copy, dedupe and hedge stages on screened dev."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(BASE / 'codex_postmortem'), str(BASE / 'codex_truck')]
from truck_replay import ap, calib, frame_rows, fusion, iou, read, screened
from audit import add_postprocess


def stage(pred, truth):
    return dict(truck=ap(frame_rows(pred, truth, 'truck')),
                genuine=ap(frame_rows(pred, truth, 'truck', genuine=True)),
                counts=dict(Counter(d['source'] for boxes in pred.values() for d in boxes)))


def diagnose(pred, truth):
    labels, ranking = defaultdict(list), []
    for frame, objects in sorted(truth.items()):
        gt = [g for g in objects if g['class'] == 'truck']
        dets = sorted(pred[frame], key=lambda d: -d['score'])
        m = iou([d['box'] for d in dets], [g['box'] for g in gt])
        vc = iou([d['box'] for d in dets], [g['box'] for g in objects if g['class'] == 'vehicle'])
        taken = set()
        for row, d in enumerate(dets):
            best = max((j for j in range(len(gt)) if j not in taken),
                       key=lambda j: m[row, j], default=None)
            if best is not None and m[row, best] >= .5:
                taken.add(best)
                outcome = 'TP'
            elif m[row].max(initial=0) >= .5:
                outcome = 'duplicate'
            elif vc[row].max(initial=0) >= .5:
                outcome = 'vehicle_overlap'
            else:
                outcome = 'other_FP'
            ranking.append((d['score'], outcome, d['source']))
        for j, obj in enumerate(gt):
            overlaps = m[:, j]
            matching = [d for value, d in zip(overlaps, dets) if value >= .5]
            labels[obj['type']].append((overlaps.max(initial=0), matching[0]['source'] if matching else '',
                                        matching[0]['score'] if matching else 0, obj['box'][3:6], obj['points']))
    summary = {}
    for subtype, rows in labels.items():
        hits = [r for r in rows if r[0] >= .5]
        summary[subtype] = dict(
            n=len(rows), matchable=len(hits), loose=int(sum(.1 <= r[0] < .5 for r in rows)),
            best_ranked_match=dict(Counter(r[1] for r in hits)),
            median_score=float(np.median([r[2] for r in hits])) if hits else None,
            label_median_dimensions=np.median([r[3] for r in rows], axis=0).round(2).tolist(),
            label_median_points=float(np.median([r[4] for r in rows])),
        )
    # Fixed budget: one prediction per labelled truck, over the whole dataset.
    top = sorted(ranking, key=lambda r: -r[0])[:sum(map(len, labels.values()))]
    return dict(subtypes=summary, top_n=len(top),
                highest_scoring_predictions=dict(Counter(f'{outcome}/{source}' for _, outcome, source in top)))


def main(size):
    screened()
    dataset = BASE / 'coop' / f'dev{size}'
    gt = {f: [g for g in gs if g['points'] >= 5] for f, gs in read(dataset / 'gt_coop.json').items()}
    pre = read(dataset / 'pred_v12p3x3_norule.json')
    assert set(pre) == set(gt)
    tagged = {f: [{**d, 'source': 'native' if d['class'] == 'truck' else 'copy'}
                  for d in rows if d['class'] in ('vehicle', 'truck')] for f, rows in pre.items()}
    del pre
    native = {f: [d for d in rows if d['class'] == 'truck'] for f, rows in tagged.items()}
    copied = calib.truck_rule(tagged, **{**calib.TRUCK, 'a': .5, 'base': .1})
    copied = {f: [d for d in rows if d['class'] == 'truck'] for f, rows in copied.items()}
    unique = calib.truck_dedupe(copied)
    baseline = add_postprocess(unique)
    for name, pred in [('native', native), ('native_dedupe_hedge', add_postprocess(calib.truck_dedupe(native))),
                       ('copies_added', copied), ('copies_and_dedupe', unique), ('final', baseline)]:
        print(json.dumps({'set': size, 'stage': name, **stage(pred, gt)}, sort_keys=True), flush=True)
    for rows in baseline.values():
        for d in rows:
            if d['hedge']:
                d['source'] += '_hedge'
    print(json.dumps({'set': size, **diagnose(baseline, gt)}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
