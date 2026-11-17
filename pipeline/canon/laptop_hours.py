"""Sum the laptop training time from the OpenPCDet training logs (last 'Train:' line of each log: total elapsed in the second bracket).
Also lists the span between the first and last training-log modification times (wall-clock window) and counts runs that overlapped."""
import re
from datetime import datetime
from pathlib import Path

L = Path.home() / 'projects/detector-program'
rx = re.compile(r'\[(\d+):(\d+):(\d+)/[^\]]*\]\s+Acc_iter')
rows = []
for f in sorted(L.glob('*.log')):
    last = None
    try:
        for line in f.open(errors='ignore'):
            if 'Train:' in line:
                last = line
    except Exception:
        continue
    if not last:
        continue
    m = re.search(r'\[(\d+):(\d{2}):(\d{2})/(\d+:\d{2}:\d{2})\]', last) or re.search(r'\[(\d+):(\d{2}):(\d{2})/', last)
    mt = re.search(r'Time cost: [^\[]*\[(\d+):(\d{2}):(\d{2})/', last)
    if not mt:
        continue
    h = int(mt.group(1)) + int(mt.group(2)) / 60 + int(mt.group(3)) / 3600
    rows.append((f.name, h, datetime.fromtimestamp(f.stat().st_mtime)))
rows.sort(key=lambda r: r[2])
tot = 0
for n, h, t in rows:
    tot += h
    print(f'{n:44s} {h:6.2f} h   ended {t:%m-%d %H:%M}')
print(f'\nruns {len(rows)} | sum of training time {tot:.1f} h')
print('first log end', rows[0][2], '| last', rows[-1][2])
