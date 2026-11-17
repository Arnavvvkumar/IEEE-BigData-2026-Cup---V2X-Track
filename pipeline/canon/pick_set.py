"""Print the dataset name of the better SECOND v3 arm (best selection-half mAP over epochs and ground heights)."""
import json
from pathlib import Path

res = Path(__file__).with_name('results')
best = {}
for tag, ds in (('canon_carlaonly_s1234', 'claude_canon_carlaonly_v1'), ('canon_all_s1234', 'claude_canon_all_v1')):
    v = [r['sel'] for p in res.glob(f'{tag}_ep*.json') for r in json.loads(p.read_text())['results'].values()]
    best[ds] = max(v) if v else -1
print(max(best, key=best.get))
