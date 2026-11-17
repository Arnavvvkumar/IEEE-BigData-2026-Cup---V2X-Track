"""Ground-truth diagnostic only: perfect removal of unmatchable truck copies."""
import json
import sys

from ray_truck import BASE, add_postprocess, ap, copy_rule, frame_rows, iou, read, screened


def run(size):
    directory = BASE / 'coop' / f'dev{size}'
    truth = {f: [g for g in rows if g['points'] >= 5]
             for f, rows in read(directory / 'gt_coop.json').items()}
    pred = add_postprocess(copy_rule(read(directory / 'pred_v12p3x3_norule.json')))
    assert len(pred) == size and pred.keys() == truth.keys()
    oracle = {}
    for frame, rows in pred.items():
        boxes = [g['box'] for g in truth[frame] if g['class'] == 'truck']
        copies = [d for d in rows if d['vehicle_copy']]
        overlaps = iou([d['box'] for d in copies], boxes)
        keep = iter(overlaps.max(axis=1, initial=0) >= .5)
        oracle[frame] = [d for d in rows if not d['vehicle_copy'] or next(keep)]
    results = {'set': size, 'removed': sum(map(len, pred.values())) - sum(map(len, oracle.values()))}
    for metric, genuine in [('all', False), ('genuine', True)]:
        results[metric] = {name: ap(frame_rows(detections, truth, 'truck', genuine=genuine))
                           for name, detections in [('baseline', pred), ('oracle', oracle)]}
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    screened()
    for arg in sys.argv[1:]:
        run(int(arg))
