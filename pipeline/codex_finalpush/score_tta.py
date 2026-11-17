"""Score one TTA policy on cached dev predictions with the other classes pinned."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'codex_postmortem'))
from truck_replay import ap, calib, finalise, frame_rows, fusion, read, screened


def ped_passes(path, ids):
    for pred in read(path).values():
        if set(pred) != set(ids):
            raise ValueError(f'Frame IDs do not match screened labels: {path}')
        yield {f: [d for d in rows if d['class'] == 'pedestrian'] for f, rows in pred.items()}


def main(size, extra):
    screened()
    dev = ROOT / 'coop' / f'dev{size}'
    run = 'v12p3x3_norule'
    spec = next(l.split(' | ')[1].removesuffix(' norule') for l in
                reversed((dev / 'eval_summary.txt').read_text().splitlines())
                if l.startswith(f'{run} | '))
    gt = {f: [d for d in rows if d['points'] >= 5] for f, rows in read(dev / 'gt_coop.json').items()}
    ids = sorted(gt)
    baseline = read(dev / f'pred_{run}.json')
    fixed, _ = finalise(baseline)
    measurements = {c: ap(frame_rows(fixed, gt, c)) for c in ('vehicle', 'pedestrian', 'truck')}
    measurements['genuine_truck'] = ap(frame_rows(fixed, gt, 'truck', genuine=True))
    print(json.dumps({'set': size, 'mode': 'baseline', **measurements}), flush=True)
    passes, missing = [], 0
    for token in spec.split():
        tag, cls = token.split(':')
        if tag.startswith('second_twin_ll_'):
            raise ValueError('Unexpected long-low voter')
        if 'pedestrian' in cls.split(','):
            passes.extend(ped_passes(dev / 'allpred' / f'{tag}_allpred.json', ids))
        else:
            missing += 6
    if len(passes) + missing != 178:
        raise ValueError('Unexpected cached voter pass count')
    for path in extra:
        passes.extend(ped_passes(path, ids))
    fusion.C = ['pedestrian']
    estimates = fusion.fuse(passes + [dict.fromkeys(ids, ())] * missing, ids)
    clouds = fusion.load(dev / 'scans')
    attenuated, _ = fusion.static_down(estimates, clouds, exclude=set(ids))
    calibrated = calib.hedge_pedestrians(calib.calibrate(attenuated, clouds))
    measurements['pedestrian'] = ap(frame_rows(calibrated, gt, 'pedestrian'))
    print(json.dumps({'set': size, 'mode': 'pedestrian_TTA', 'paths': list(map(str, extra)),
                      **measurements}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('size', type=int, choices=(150, 350))
    parser.add_argument('extra', type=Path, nargs='+')
    args = parser.parse_args()
    main(args.size, args.extra)
