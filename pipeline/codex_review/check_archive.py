"""Read-only Track B archive checks, stricter than the scorer's text/JSON checks."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from replay import EXP, KIT, read, screened, spec, P1, P2, V1

DECL = ('I confirm that I did not train on any frame listed in forbidden_frames.txt, '
        'and that all submitted detections were produced by a model trained only on '
        'synthetic data I generated.')
CLASSES = ('vehicle','pedestrian','truck')


def no_duplicates(pairs):
    d = dict(pairs)
    if len(d) != len(pairs):
        raise ValueError('duplicate JSON key')
    return d


def verify(path):
    ids = read(KIT / 'detection_test_frame_ids.json')
    cfg = read(KIT / 'reference_data_dummy/config.json')
    assert set(cfg['class_names']) == set(CLASSES)
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    with ZipFile(path) as z:
        bins = sorted(n for n in z.namelist() if re.fullmatch(r'synthetic/[^/]+\.bin', n))
        check(len(bins) == 50, 'expected 50 synthetic files')
        check(set(z.namelist()) == set(bins) | {'declaration.txt','predictions.json'}, 'unexpected ZIP entry')
        check(len(set(z.namelist())) == len(z.namelist()), 'duplicate ZIP entry')
        check(z.testzip() is None, 'CRC check failed')
        total = sum(x.file_size for x in z.infolist())
        check(total < 75_000_000, 'total size cap exceeded')
        syn = EXP / 'twin/final_synthetic_v4/synthetic'
        v4 = True
        largest = 0
        for n in bins:
            data = z.read(n)
            largest = max(len(data), largest)
            check(len(data) > 0 and len(data) <= 1_500_000 and len(data) % 12 == 0
                  and np.isfinite(np.frombuffer(data,dtype='<f4')).all(), f'bad XYZ {n}')
            actual = syn / Path(n).name
            v4 &= actual.is_file() and data == actual.read_bytes()
        decl = z.read('declaration.txt').decode('utf-8-sig')
        check(DECL.casefold() in ' '.join(decl.split()).casefold(), 'full required declaration text missing')
        for label in ('Team / individual name','Date (YYYY-MM-DD)','Submission tag'):
            match = re.search(r'^'+re.escape(label)+r':\s*([^\n]*)',decl,re.M)
            val = match[1].strip() if match else ''
            check(bool(val and val.strip('_')), 'missing ' + label)
        p = json.loads(z.read('predictions.json'), object_pairs_hook=no_duplicates)
        check(set(p) == set(ids), 'incorrect frame ID set')
        count = 0
        for ds in p.values():
            for d in ds:
                count += 1
                b,s = d['box'],d['score']
                if not (len(b)==7 and d['class'] in CLASSES and all(map(math.isfinite,b))
                        and min(b[3:6]) > 0 and math.isfinite(s) and 0 <= s <= 1
                        and abs(b[6]) <= math.pi + 1e-6):
                    errors.append('invalid detection')
                    break
    return {'archive':path.name,'uncompressed_bytes':total,'largest_xyz_bytes':largest,
            'synthetic_files':len(bins),'v4_bytes_match':bool(v4),'frame_count':len(p),
            'kit_frame_order':list(p)==ids,'lexical_frame_order':list(p)==sorted(ids),
            'detections':count,'full_declaration': DECL.casefold() in ' '.join(decl.split()).casefold(),
            'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest')
                     else hashlib.sha256(path.read_bytes()).hexdigest(),'errors':errors}


if __name__ == '__main__':
    ref = screened()
    expected = {n for n,_ in spec()} | {P1,P2,V1}
    passed = EXP / 'final/test_allpred_ps'
    present = {p.name.removesuffix('_allpred.json') for p in passed.glob('*_allpred.json')}
    log = (EXP/'final/reexport_perscan.log').read_text()
    n_logged = log.count('plane mode per-scan')
    print(json.dumps({'reference_forbidden_overlap':0,'expected_export_tags':len(expected),'present':len(present),
                      'missing_filenames':sorted(expected-present), 'per_scan_modes_logged':n_logged}))
    failed = False
    for raw in sys.argv[1:]:
        row = verify(Path(raw))
        failed |= bool(row['errors'])
        print(json.dumps(row),flush=True)
    sys.exit(int(failed))
