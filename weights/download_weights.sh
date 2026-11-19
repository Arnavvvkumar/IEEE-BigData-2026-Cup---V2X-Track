#!/bin/bash
# Download the 18 final-ensemble checkpoints from the GitHub release and verify their SHA-256 hashes.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p ckpt
BASE="https://github.com/Arnavvvkumar/IEEE-BigData-2026-Cup---V2X-Track/releases/download/v1.0"
python3 - <<'PY' > /tmp/bdc_weights.txt
import json
for m in json.load(open('manifest.json')): print(m['file'], m['sha256'])
PY
while read -r f sha; do
  [ -f "ckpt/$f" ] || curl -fL --retry 3 -o "ckpt/$f" "$BASE/$f"
  echo "$sha  ckpt/$f" | sha256sum -c -
done < /tmp/bdc_weights.txt
echo "all checkpoints downloaded and verified in weights/ckpt/"
