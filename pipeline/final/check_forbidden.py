#!/usr/bin/env python
"""Refuse a validation folder if any frame is one of the 100 forbidden frames.
usage: check_forbidden.py <set dir with index.json> ...
"""
import os
import csv
import json
import sys
from pathlib import Path

W = Path(os.environ['BDC_ROOT'])
forbidden = set()
with (W / 'experiments/real_reference_audit/v2x_real_forbidden_frames_mapping.csv').open(newline='') as handle:
    for row in csv.DictReader(handle):
        forbidden.add(row['point_cloud_file'])
failed = False
for raw in sys.argv[1:]:
    folder = Path(raw)
    index = json.loads((folder / 'index.json').read_text())
    bad = [name for name, meta in index.items() if meta['source_key'] + '.bin' in forbidden]
    print(folder.name, 'frames', len(index), 'forbidden', len(bad))
    if bad:
        print('overlap', bad[:10])
        failed = True
if failed:
    raise SystemExit(1)
