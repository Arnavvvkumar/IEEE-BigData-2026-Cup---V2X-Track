#!/bin/bash
# READ-ONLY on OSC: find truncated .npy files (header shape vs file size) and missing label files in the three training sets.
# Writes the bad list locally to osc/bad_files.txt (paths relative to data/).
here="${BDC_ROOT}/experiments/claude_carla_overnight/osc"
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>/dev/null << 'EOF' > "$here/bad_scan.out"
R=/fs/scratch/${OSC_PROJECT}/bdc
source /etc/profile 2>/dev/null
module load python/3.10 2>/dev/null
source $R/env/bin/activate
python - << 'PY'
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
D = Path('/fs/scratch/${OSC_PROJECT}/bdc/data')
def check(p):
    try:
        with open(p, 'rb') as f:
            v = np.lib.format.read_magic(f)
            shape, _, dt = np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f)
            need = f.tell() + int(np.prod(shape)) * dt.itemsize
        return None if os.path.getsize(p) == need else p
    except Exception:
        return p
for s in ('claude_canon_all_v1', 'claude_twin_v1', 'claude_twin_st_v1'):
    files = sorted((D / s / 'points').glob('*.npy'))
    with ThreadPoolExecutor(16) as ex:
        bad = [b for b in ex.map(check, files) if b]
    ids = (D / s / 'ImageSets/train.txt').read_text().split() + ((D / s / 'ImageSets/val.txt').read_text().split() if (D / s / 'ImageSets/val.txt').exists() else [])
    miss_p = [i for i in ids if not (D / s / 'points' / f'{i}.npy').exists()]
    miss_l = [i for i in ids if not (D / s / 'labels' / f'{i}.txt').exists() or (D / s / 'labels' / f'{i}.txt').stat().st_size == 0 and False]
    print(f'SUMMARY {s}: files {len(files)} bad {len(bad)} missing_points {len(miss_p)} missing_labels {len(miss_l)}')
    for b in bad:
        print('BAD', b.relative_to(D))
    for i in miss_p:
        print('BAD', f'{s}/points/{i}.npy')
    for i in miss_l:
        print('BAD', f'{s}/labels/{i}.txt')
PY
EOF
grep SUMMARY "$here/bad_scan.out"
grep '^BAD ' "$here/bad_scan.out" | cut -d' ' -f2 | sort -u > "$here/bad_files.txt"
echo "bad files: $(wc -l < "$here/bad_files.txt")"; head -n 5 "$here/bad_files.txt"
cd ${BDC_DATA} && echo "local size of bad files: $(xargs -a "$here/bad_files.txt" -r du -cL 2>/dev/null | tail -n 1)"
