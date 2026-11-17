"""Verify a frozen candidate archive using the original builder and kit scorer."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from check_archive import CLASSES, verify
from replay import EXP, KIT, P1, screened, spec
import numpy as np


def compare(candidate, reference):
    for cls in CLASSES:
        rows = []
        for pred in (reference, candidate):
            scores = [d['score'] for ds in pred.values() for d in ds if d['class'] == cls]
            rows.append({'count': len(scores),
                         'quantiles': np.quantile(scores, [0, .25, .5, .75, 1]).tolist() if scores else []})
        print(json.dumps({'class': cls, 'v10r': rows[0], 'candidate': rows[1]}), flush=True)


def rebuild(work, extra, export_dir):
    final = EXP / 'final'
    builder = (final / 'build_candidate_v7.sh').read_text()
    postproc = builder.split("<< 'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    scans = KIT / 'detection_test_frames'
    env = os.environ.copy()
    for key in ('FUSE_TRUCK_SCORE', 'TRUCK_A', 'TRUCK_BASE'):
        env.pop(key, None)
    env.update(FUSE_YAW_CIRC='vehicle', FUSE_ONE_VOTE='vehicle',
               TRUCKPP='1', TRUCKBOOST='1.0', HEDGE='1')
    voters = spec() + [(P1, 'pedestrian')] * 3
    for voter in extra:
        model, classes = voter.split(':')
        if not re.fullmatch(r'[A-Za-z0-9_]+', model) or model.startswith('second_twin_ll_'):
            raise ValueError(f'invalid model {model}')
        if not set(classes.split(',')) <= set(CLASSES):
            raise ValueError(f'invalid classes {classes}')
        voters.append((model, classes))
    fused, output = work / 'fused.json', work / 'predictions.json'
    args = [f'{export_dir}/{tag}_allpred.json::{cls}' for tag, cls in voters]
    subprocess.run([sys.executable, '-B', str(final / 'fuse_test.py'),
                    str(scans), str(fused), *args], check=True, env=env)
    subprocess.run([sys.executable, '-B', '-c', postproc,
                    str(scans), str(fused), str(output)], check=True, cwd=final, env=env)
    return output


def official_validate(work, archive, dev):
    kit = work / 'kit'
    kit.mkdir()
    shutil.copy2(KIT / 'local_eval.py', kit)
    shutil.copytree(KIT / 'scoring_program', kit / 'scoring_program')
    ref = kit / 'reference_data_dummy'
    shutil.copytree(KIT / 'reference_data_dummy', ref)
    realism = ref / 'realism_reference'
    shutil.rmtree(realism)
    realism.mkdir()
    clouds = sorted((dev / 'detection_test_frames').glob('*.bin'))
    if len(clouds) != 50:
        raise ValueError('expected 50 screened reference scans')
    for cloud in clouds:
        shutil.copy2(cloud, realism)
    cfg = ref / 'config.json'
    options = json.loads(cfg.read_text())
    options['n_realism_frames'] = 50
    cfg.write_text(json.dumps(options))
    env = os.environ.copy()
    deps = Path.home() / 'projects/roadside-pilot/realism_deps'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(deps), env.get('PYTHONPATH', ''))))
    subprocess.run([sys.executable, '-B', str(kit / 'local_eval.py'), str(archive), str(ref)],
                   check=True, env=env)
    valid = json.loads((kit / 'scores.json').read_text()).get('valid_submission') == 1
    print(json.dumps({'kit_valid_submission': valid, 'reference': 'screened dev and dummy GT'}))
    return valid


def main():
    archive = Path(sys.argv[1]).resolve()
    validation_only = sys.argv[2:] == ['--kit-only']
    dev = screened()
    audit = verify(archive)
    print(json.dumps(audit), flush=True)
    if audit['errors'] or not audit['v4_bytes_match']:
        return 1
    with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        work = Path(tmp)
        if validation_only:
            return 0 if official_validate(work, archive, dev) else 1
        with ZipFile(archive) as z, ZipFile(Path(sys.argv[2]).resolve()) as baseline:
            original = z.read('predictions.json')
            compare(json.loads(original), json.loads(baseline.read('predictions.json')))
        export_dir = Path(os.environ.get('AUDIT_EXPORT_DIR', EXP / 'final/test_allpred_ps'))
        print(json.dumps({'export_directory': str(export_dir), 'extra_voters': sys.argv[3:]}), flush=True)
        output = rebuild(work, sys.argv[3:], export_dir).read_bytes()
        same_bytes = output == original
        same_predictions = same_bytes or json.loads(output) == json.loads(original)
        print(json.dumps({'prediction_bytes_equal': same_bytes, 'predictions_equal': same_predictions}), flush=True)
        valid = official_validate(work, archive, dev)
    return 0 if valid and same_predictions else 1


if __name__ == '__main__':
    sys.exit(main())
