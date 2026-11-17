"""Compare batch-1 vs batch-4 predictions as SETS per frame (order-insensitive): for each box in one file, nearest same-class
box in the other file of the same frame; report the worst mismatch. Also checks frames are not swapped."""
import json
from pathlib import Path

import numpy as np

t = Path('/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/coop/batchcheck')
a, b = json.loads((t / 'b1.json').read_text()), json.loads((t / 'b4.json').read_text())
worst_box = worst_score = 0.0
order_same = 0
total = 0
for g in a:
    for f in a[g]:
        x, y = a[g][f], b[g][f]
        X = np.array([p['box'] + [p['score']] for p in x]).reshape(-1, 8)
        Y = np.array([p['box'] + [p['score']] for p in y]).reshape(-1, 8)
        cx = [p['class'] for p in x]
        cy = [p['class'] for p in y]
        total += 1
        order_same += all(np.allclose(p['box'], q['box'], atol=1e-4) for p, q in zip(x, y))
        for i in range(len(X)):
            m = [j for j in range(len(Y)) if cy[j] == cx[i]]
            if not m:
                worst_box = float('inf')
                continue
            d = np.abs(Y[m, :7] - X[i, :7]).max(1)
            j = int(d.argmin())
            worst_box = max(worst_box, float(d[j]))
            worst_score = max(worst_score, abs(float(Y[m[j], 7] - X[i, 7])))
print('frames', total, '| same order', order_same, '| worst nearest-box diff', round(worst_box, 6), '| worst score diff', round(worst_score, 6))
