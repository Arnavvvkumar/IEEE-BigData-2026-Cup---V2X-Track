"""Score a one-checkpoint levelling ablation using production postprocessing."""

import sys
from pathlib import Path

from truck_replay import BASE, calib, fusion, read, screened, summary


def main():
    screened()
    size, out = int(sys.argv[1]), Path(sys.argv[2])
    dev = BASE / 'coop' / f'dev{size}'
    clouds = fusion.load(out / 'scans' if (out / 'scans').exists() else dev / 'scans')
    labels = read(dev / 'gt_coop.json')
    truth = {f: [g for g in labels[f] if g['points'] >= 5] for f in clouds}
    fusion.C = ['vehicle', 'truck']
    fusion.ONE_VOTE = {'vehicle'}
    fusion.YAW_CIRC = {'vehicle'}
    for name in ('median', 'independent'):
        data = read(out / f'{name}.json')
        assert all(set(p) == set(clouds) for p in data.values())
        pred = calib.calibrate(fusion.fuse(list(data.values()), sorted(truth)), clouds)
        summary(size, name, pred, truth, pedestrian=[])


if __name__ == '__main__':
    main()
