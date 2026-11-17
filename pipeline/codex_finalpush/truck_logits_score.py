"""Measure the class-score export diagnostic, keeping stock vehicle boxes."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'codex_postmortem'), str(ROOT / 'codex_truck')]
from truck_replay import ap, calib, frame_rows, iou, read, screened
from audit import add_postprocess


def main(size, out):
    screened()
    gt = read(ROOT / 'coop' / f'dev{size}' / 'gt_coop.json')
    stock, extra = (read(out / f'{mode}.json')['-3.0'] for mode in ('stock', 'retain'))
    assert set(stock) == set(extra)
    gt = {f: [g for g in gt[f] if g['points'] >= 5] for f in sorted(stock)}
    for mode, p in [('stock', stock), ('retain', extra)]:
        native = {f: [d for d in p[f] if d['class'] == 'truck'] for f in gt}
        counts = defaultdict(lambda: [0, 0])
        for f, rows in gt.items():
            labels = [g for g in rows if g['class'] == 'truck']
            matches = iou([d['box'] for d in native[f]], [g['box'] for g in labels]).max(0, initial=0)
            for g, match in zip(labels, matches):
                bucket = counts[g['type']]
                bucket[0] += 1
                bucket[1] += int(match >= .5)
        merged = {f: native[f] + [d for d in stock[f] if d['class'] == 'vehicle'] for f in gt}
        clouds = {f: np.empty((0, 3)) for f in gt}
        calibrated = calib.calibrate(merged, clouds)
        ruled = calib.truck_rule(calibrated, **{**calib.TRUCK, 'a': .5, 'base': .1})
        trucks = {f: [d for d in rows if d['class'] == 'truck'] for f, rows in ruled.items()}
        processed = add_postprocess(calib.truck_dedupe(trucks))
        print(json.dumps({'set': size, 'mode': mode, 'native_n_and_IoU50': counts,
                          'truck_AP': ap(frame_rows(processed, gt, 'truck')),
                          'genuine_AP': ap(frame_rows(processed, gt, 'truck', True))}), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), Path(sys.argv[2]))
