#!/bin/bash
# Does a harder compressor shrink these clouds more than gzip -1?
python3 - << 'PY'
import gzip, os, glob
path = sorted(glob.glob(os.path.join(os.environ["BDC_DATA"], "claude_twin_v1/points/twin_*.npy")))[0]
data = open(path, "rb").read()
for level in (1, 6, 9):
    out = gzip.compress(data, compresslevel=level)
    print(os.path.basename(path), "level", level, "ratio", round(len(out) / len(data), 3), "bytes", len(data))
PY
