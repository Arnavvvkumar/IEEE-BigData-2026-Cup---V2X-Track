"""Splice: vehicle+truck detections from a reference zip (v13, proven), pedestrian detections from a new zip. Synthetic + declaration from new zip.
usage: splice_ped.py <ref.zip> <ped.zip> <out.zip> <tag>"""
import json, sys, zipfile, collections
ref, ped, out, tag = sys.argv[1:5]
A, B = zipfile.ZipFile(ref), zipfile.ZipFile(ped)
pa, pb = json.loads(A.read('predictions.json')), json.loads(B.read('predictions.json'))
assert type(pa) is type(pb)

def dets(v):
    return v if isinstance(v, list) else None

def splice(a, b):
    if isinstance(a, dict) and all(isinstance(v, list) for v in a.values()):
        assert set(a) == set(b), 'frame sets differ'
        return {f: [x for x in a[f] if x['class'] != 'pedestrian'] + [x for x in b[f] if x['class'] == 'pedestrian'] for f in a}
    raise SystemExit(f'unexpected structure: {type(a)} keys {list(a)[:5] if isinstance(a, dict) else None}')

pc = splice(pa, pb)
cnt = lambda p: dict(collections.Counter(x['class'] for v in p.values() for x in v))
print('ref', cnt(pa), '| ped src', cnt(pb), '| out', cnt(pc))
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as Z:
    for n in B.namelist():
        if n == 'predictions.json':
            Z.writestr(n, json.dumps(pc))
        elif n == 'declaration.txt':
            Z.writestr(n, B.read(n).decode().replace('v14-final', tag))
        else:
            Z.writestr(B.getinfo(n), B.read(n))
C = zipfile.ZipFile(out)
pv = json.loads(C.read('predictions.json'))
print('non-ped identical to ref:', all([x for x in pv[f] if x['class'] != 'pedestrian'] == [x for x in pa[f] if x['class'] != 'pedestrian'] for f in pa))
print('ped identical to src:', all([x for x in pv[f] if x['class'] == 'pedestrian'] == [x for x in pb[f] if x['class'] == 'pedestrian'] for f in pb))
print([l for l in C.read('declaration.txt').decode().splitlines() if l.startswith(('Date', 'Submission'))])
