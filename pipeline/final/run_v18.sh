#!/bin/bash
# v18s = v17s + truck min-points rule (truck boxes with < 30 non-ground scan points inside -> score x0.25, before the truck size hedge).
# Vehicle boxes stay byte-identical to v13/v17s, pedestrian boxes byte-identical to v17s. Declaration date corrected to 2026-11-15.
# Also writes v17s_d15 (v17s with the corrected declaration only). Builds locally; nothing is submitted.
set -uo pipefail
F="${BDC_ROOT}/experiments/claude_carla_overnight/final"
cd ${BDC_OUT}
sed -i 's/\r$//' "$F/build_v13.sh" "$F/build_candidate_v7.sh" "$F/run_splice.sh"
# 1) v13 recipe + truck rule  -> v13t (only its vehicle/truck boxes are used)
export TRUCK_MINPTS=1
ADD='pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian pvrcnn_ped_P3_ep5:pedestrian' bash "$F/build_v13.sh" 13t
unset TRUCK_MINPTS
ls -la bdc2026_trackB_submission_v13t.zip || exit 1
# 2) splice: vehicle+truck from v13t, pedestrian from v17
python3 "$F/splice_ped.py" bdc2026_trackB_submission_v13t.zip bdc2026_trackB_submission_v17.zip _v18s_tmp.zip v18s-final
# 3) declaration fix (date + tag) for v18s and v17s
python3 - <<'EOF'
import re, zipfile
def fix(src, dst, tag):
    A = zipfile.ZipFile(src)
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as Z:
        for n in A.namelist():
            if n == 'declaration.txt':
                t = A.read(n).decode()
                t = re.sub(r'^Date \(YYYY-MM-DD\):.*$', 'Date (YYYY-MM-DD):      2026-11-15', t, flags=re.M)
                t = re.sub(r'^Submission tag:.*$', 'Submission tag:         ' + tag, t, flags=re.M)
                Z.writestr(n, t)
            else:
                Z.writestr(A.getinfo(n), A.read(n))
    print(dst, [l for l in zipfile.ZipFile(dst).read('declaration.txt').decode().splitlines() if l.startswith(('Date', 'Submission'))])
fix('_v18s_tmp.zip', 'bdc2026_trackB_submission_v18s.zip', 'v18s-final-2026-11-15')
fix('bdc2026_trackB_submission_v17s.zip', 'bdc2026_trackB_submission_v17s_d15.zip', 'v17s-final-2026-11-15')
EOF
rm -f _v18s_tmp.zip
for z in v18s v17s_d15; do echo "== kit $z"; bash "$F/kit_validate.sh" bdc2026_trackB_submission_$z.zip 2>&1 | grep -E 'valid_submission|score_realism'; done
# 4) what changed vs v17s / v13
python3 - <<'EOF'
import json, zipfile, collections
L = lambda z: json.loads(zipfile.ZipFile(f'bdc2026_trackB_submission_{z}.zip').read('predictions.json'))
a, b, c = L('v17s'), L('v18s'), L('v13')
cls = lambda p, k: {f: [x for x in v if x['class'] == k] for f, v in p.items()}
for k in ('vehicle', 'pedestrian'):
    print(k, 'v18s identical to v17s:', cls(a, k) == cls(b, k))
key = lambda x: tuple(round(v, 5) for v in x['box'])
same = scaled = other = missing = added = 0
nt0 = nt1 = 0
for f in c:
    old = collections.defaultdict(list); new = collections.defaultdict(list)
    for x in cls(c, 'truck')[f]: old[key(x)].append(x['score'])
    for x in cls(b, 'truck')[f]: new[key(x)].append(x['score'])
    nt0 += sum(map(len, old.values())); nt1 += sum(map(len, new.values()))
    for k_, s in old.items():
        if k_ not in new: missing += len(s); continue
        for s0, s1 in zip(sorted(s), sorted(new[k_])):
            r = s1 / s0 if s0 else 1
            same += abs(r - 1) < 1e-6; scaled += abs(r - 0.25) < 1e-4; other += not (abs(r - 1) < 1e-6 or abs(r - 0.25) < 1e-4)
    added += sum(1 for k_ in new if k_ not in old)
print(f'truck boxes v13 {nt0} -> v18s {nt1} | unchanged {same} | x0.25 {scaled} | other ratio {other} | missing (hedge copies of down-weighted parents) {missing} | new {added}')
sc = lambda p: sorted(x['score'] for v in p.values() for x in v if x['class'] == 'truck')
s0, s1 = sc(c), sc(b)
print('truck score max v13', round(s0[-1], 4), 'v18s', round(s1[-1], 4), '| top-50 sum', round(sum(s0[-50:]), 3), round(sum(s1[-50:]), 3))
EOF
echo V18_ALL_DONE
