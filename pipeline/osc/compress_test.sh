#!/bin/bash
# Compression ratio on a sample of each dataset's point files (what would travel over the slow uplink).
D="$HOME/projects/bdc2026-track-b/data"
for ds in claude_twin_v1 claude_canon_all_v1; do
  files=$(ls -L "$D/$ds/points" | head -n 3000 | awk 'NR%30==0' | head -n 60)
  raw=0; gz=0; zs=0
  tmp=$(mktemp -d)
  for f in $files; do cp -L "$D/$ds/points/$f" "$tmp/$f"; done
  raw=$(du -cb "$tmp"/* | tail -n 1 | cut -f1)
  gz=$(tar -C "$tmp" -cf - . | gzip -1 | wc -c)
  if command -v zstd > /dev/null; then zs=$(tar -C "$tmp" -cf - . | zstd -3 -T4 -q | wc -c); fi
  echo "$ds sample $(echo $files | wc -w) files raw $((raw/1000000)) MB | gzip-1 $((gz/1000000)) MB ratio $(echo "scale=2; $gz/$raw" | bc) | zstd-3 $((zs/1000000)) MB ratio $(echo "scale=2; $zs/$raw" | bc)"
  rm -rf "$tmp"
done
python3 - <<'EOF'
import numpy as np, glob, os
D = os.path.expanduser('~/projects/bdc2026-track-b/data')
for ds in ('claude_twin_v1', 'claude_canon_all_v1'):
    f = sorted(glob.glob(f'{D}/{ds}/points/*.npy'))[7]
    a = np.load(f)
    print(ds, 'shape', a.shape, a.dtype, '| intensity unique values:', len(np.unique(a[:, 3])), '| zero-points share:', round(float((np.abs(a[:, :3]).sum(1) == 0).mean()), 3))
EOF
