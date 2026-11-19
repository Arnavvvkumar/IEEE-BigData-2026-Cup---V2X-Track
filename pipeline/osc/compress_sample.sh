#!/bin/bash
# Compare compressors on a handful of twin point clouds. Local only.
set -euo pipefail
D="${BDC_DATA}/claude_twin_v1/points"
tmp=$(mktemp -d)
i=0
for f in $(ls -L "$D" | head -n 8); do
  cp -L "$D/$f" "$tmp/$f"
  i=$((i+1))
done
raw=$(du -cb "$tmp"/* | tail -n 1 | cut -f1)
gz=$(tar -C "$tmp" -cf - . | gzip -1 | wc -c)
echo "files $i raw $raw gzip1 $gz ratio $(python3 -c "print(round($gz/$raw, 3))")"
rm -rf "$tmp"
