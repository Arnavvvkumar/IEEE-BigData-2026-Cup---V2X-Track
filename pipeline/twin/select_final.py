"""Choose the 50 submitted realism scans from the saved twin scans, check them with the UNMODIFIED kit function, export .bin.
Selection: random 50-subsets scored (kit metric, chunked MMD) against the SELECTION half of the real dev frames only;
the winner is adopted only if it also beats the arbitrary first-50 baseline on the CONFIRMATION half.
usage: select_final.py <twin dataset dir> <out dir> [trials=36] [workers=3]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import range_bg as T  # noqa: E402

RANGE = (-40.0, -40.0, -8.0, 40.0, 40.0, 2.0)
NORM = {'CD': (2.0, 30.0), 'MMD': (0.0, 0.5), 'EMD': (0.0, 5.0), 'FPD': (0.0, 50.0)}
d, out = Path(sys.argv[1]), Path(sys.argv[2])
trials = int(sys.argv[3]) if len(sys.argv) > 3 else 36
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
files = sorted((d / 'sensor').glob('*.npy'))
SEL = [T.clouds[f] for f in T.split['selection_ids']]
CONF = [T.clouds[f] for f in T.split['confirmation_ids']]


def subset(k):
    if k == 0:
        return list(range(50))
    return sorted(np.random.default_rng(7000 + k).choice(len(files), 50, replace=False).tolist())


def score_sel(k):
    idx = subset(k)
    return k, T.lean([np.load(files[i]) for i in idx], SEL)


def kit(scans, ref):
    sys.path.insert(0, str(T.W / 'starting_kit/track_b/scoring_program'))
    import metrics as M
    raw = M.compute_realism(scans, ref, RANGE, seed=1234)
    s = {k: float(np.clip((NORM[k][1] - raw[k]) / (NORM[k][1] - NORM[k][0]), 0, 1)) for k in NORM}
    return {'realism': round(float(np.mean(list(s.values()))), 4), **{k: round(float(raw[k]), 5) for k in NORM},
            **{k + '_n': round(v, 3) for k, v in s.items()}}


if __name__ == '__main__':
    with Pool(workers) as pool:
        res = dict(pool.map(score_sel, range(trials + 1), chunksize=1))
    order = sorted(res, key=lambda k: -res[k]['realism'])
    best = order[0]
    log = {'n_candidates': len(files), 'trials': trials, 'baseline_sel': res[0], 'best_trial': best, 'best_sel': res[best],
           'sel_scores_sorted': [res[k]['realism'] for k in order]}
    base_conf = T.lean([np.load(files[i]) for i in subset(0)], CONF)
    best_conf = T.lean([np.load(files[i]) for i in subset(best)], CONF)
    log.update(baseline_conf=base_conf, best_conf=best_conf)
    pick = best if best_conf['realism'] >= base_conf['realism'] else 0
    idx = subset(pick)
    scans = [np.load(files[i]).astype(np.float32) for i in idx]
    log.update(picked_trial=pick, picked_files=[files[i].name for i in idx])
    (out / 'synthetic').mkdir(parents=True)
    for j, s in enumerate(scans, 1):
        s[:, :3].astype('<f4').tofile(out / 'synthetic' / f'{j:04d}.bin')
    back = [np.fromfile(f, dtype='<f4').reshape(-1, 3) for f in sorted((out / 'synthetic').glob('*.bin'))]
    assert len(back) == 50 and all(np.array_equal(a, b) for a, b in zip(back, scans))
    log.update(kit_unmodified_vs_all50=kit(back, [T.clouds[f] for f in T.ids]), kit_unmodified_vs_confirmation=kit(back, CONF),
               bytes_max=int(max(f.stat().st_size for f in (out / 'synthetic').glob('*.bin'))),
               points_median=int(np.median([len(s) for s in back])),
               zeros_median=int(np.median([(np.linalg.norm(s, axis=1) == 0).sum() for s in back])))
    (out / 'selection_log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps({k: v for k, v in log.items() if k != 'picked_files'}, indent=1))
