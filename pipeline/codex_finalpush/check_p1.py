"""Require the same P1 weight to beat both cached dev sets and all halves."""

import ast
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text()
results = {}
for line in log.splitlines():
    match = re.search(r'^(dev150|dev350) (v13(?:_P1tta_x[13])?_norule) (PED|VT): (.*)', line)
    if not match:
        continue
    size, arm, _, summary = match.groups()
    values = {metric: ast.literal_eval(triple) for metric, triple in re.findall(
        r'(pedestrian AP|vehicle|truck|genuine) (\[[^\]]+\])', summary)}
    results.setdefault((size, arm), {}).update(values)

for arm in ('v13_P1tta_x1_norule', 'v13_P1tta_x3_norule'):
    passed = True
    for size in ('dev150', 'dev350'):
        reference = results.get((size, 'v13base_norule'), {})
        candidate = results.get((size, arm), {})
        for metric in ('pedestrian AP', 'vehicle', 'truck', 'genuine'):
            old, new = reference.get(metric, []), candidate.get(metric, [])
            passed &= len(old) == len(new) == 3 and all(
                after > before if metric == 'pedestrian AP' else after >= before
                for before, after in zip(old, new))
    if passed:
        print(f'P1 pilot gate passed for {arm}')
        sys.exit(0)
sys.exit('P1 pilot has no verified all-half pass')
