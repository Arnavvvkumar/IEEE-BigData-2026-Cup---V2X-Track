import json
import sys

d = json.load(open(sys.argv[1]))
print(d['relaxed_metrics'])
for c in ('vehicle', 'pedestrian', 'truck'):
    x = d['per_class'][c]
    print(c, x['outcomes'], x.get('poor_box_errors'))
    print(' range', x['by_range'])
    print(' pts', x['by_points'])
print(d['false_positives_score>=0.3'])
