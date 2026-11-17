"""Check the magnitude of levelling differences on screened development scans."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(EXP / 'codex_review'), str(EXP / 'erroranalysis')]
from groundplane import fit_ground, level_transform
from replay import screened


def summarize(values):
    return np.quantile(values, [0, .1, .5, .9, 1], axis=0).round(5).tolist()


def main():
    ref = screened()
    panels = [(ref / 'source_index.json', ref / 'detection_test_frames', 'local50'),
              *[(EXP / 'coop' / f'dev{n}' / 'index.json',
                 EXP / 'coop' / f'dev{n}' / 'scans', f'dev{n}') for n in (150, 350)]]
    for index_path, scans, name in panels:
        groups = defaultdict(list)
        for key, row in sorted(json.loads(index_path.read_text()).items()):
            groups[row['source_key'].split('/')[0]].append(Path(key).stem)
        # Take the middle frame in each scenario. One scene gets one vote.
        ids = [frames[len(frames) // 2] for frames in groups.values()]
        estimates, points = [], []
        for key in ids:
            path = scans / f'{key}.bin'
            if not path.is_file():
                raise FileNotFoundError(path)
            p = np.fromfile(path, dtype='<f4').reshape(-1, 3)
            p = p[np.isfinite(p).all(1)]
            keep = ((p >= [-40, -40, -8]) & (p <= [40, 40, 2])).all(1)
            q = p[keep]
            zero = (q == 0).all(1)
            ranges = np.linalg.norm(q[~zero, :2], axis=1)
            points.append([len(q), int(zero.sum()), np.mean(ranges > 35)])
            estimates.append(fit_ground(p, np.random.default_rng(1234)))
        abc = np.asarray(estimates)
        xyz = np.array([[37.5, 0, -3], [-37.5, 0, -3], [0, 37.5, -3], [0, -37.5, -3]])
        r, t = level_transform(np.median(abc, axis=0), -3)
        med = (xyz @ r.T + t)[:, 2]
        deviations = []
        for plane in abc:
            rotation, translation = level_transform(plane, -3)
            deviations.append(abs((xyz @ rotation.T + translation)[:, 2] - med).max())
        print(json.dumps({'panel': name, 'scenarios': len(ids), 'quantiles': [0, .1, .5, .9, 1],
                          'height_then_tilt_deg': summarize(np.c_[-abc[:, 2], np.degrees(np.arctan(np.hypot(abc[:, 0], abc[:, 1])))]),
                          'crop_rows_origin_rows_nonzero_xy_fraction_over_35m': summarize(points),
                          'max_z_delta_m_at_37p5m_vs_panel_median': summarize(deviations)}), flush=True)


if __name__ == '__main__':
    main()
