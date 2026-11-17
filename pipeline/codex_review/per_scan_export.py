"""Run the unchanged exporter with deterministic ground fitting and one scan per batch."""

import os
import runpy
import sys
from pathlib import Path

import numpy as np


def fit_independently(fit_ground, clouds, ids, workers):
    return np.asarray([fit_ground(clouds[frame], np.random.default_rng(1234)) for frame in ids])


def main():
    final = Path(__file__).resolve().parent.parent / 'final'
    mode, *args = sys.argv[1:]
    if mode not in ('infer_test.py', 'infer_tta.py'):
        raise ValueError(f'unknown exporter {mode}')
    repo = Path(args[args.index('--repo') + 1]).resolve()
    sys.path[:0] = [str(final), str(repo)]
    import batched_forward

    batched_forward.fit_planes = fit_independently
    os.environ['PLANE'] = 'scan'
    sys.argv = [str(final / mode), *args, '--batch', '1']
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
