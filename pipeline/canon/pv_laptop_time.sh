#!/bin/bash
# Past PV-RCNN / SECOND timings on this laptop (from training logs), and the GPU model.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd ${BDC_PROGRAM}
for f in $(ls *.log 2>/dev/null | grep -i -E 'pvrcnn|pv_rcnn|second_canon_all_s1234|second_twin_s1|vnx010_twin_s1\.log' | head -n 8); do
  n=$(grep -c 'Train:' "$f")
  echo "== $f ($n progress lines)"
  grep -E 'Train:' "$f" | tail -n 1 | sed -E 's/.*Train: +//' | cut -c1-205
  grep -E 'Train:' "$f" | sed -n '20p' | sed -E 's/.*(Batch time: [0-9.()]+).*/early \1/'
done
