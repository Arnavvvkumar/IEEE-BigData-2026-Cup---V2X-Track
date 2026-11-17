#!/bin/bash
# Build + audit + Codex-verify a mix from a converted CARLA dir. Usage: mix_test.sh <converted_dir> <out_dir>
set -euo pipefail
cd "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
E=~/projects/bdc2026-track-b/.env/bin/python
BASE=~/projects/bdc2026-track-b/data/codex_openpcdet_custom_v1
CANON=~/projects/bdc2026-track-b/data/claude_v2_scale5500_fix1/frames.json
CONV="$1"; OUT="$2"
nice -n 10 taskset -c 8-13 $E -I build_mix.py --base $BASE --carla "$CONV" --out "$OUT"
nice -n 10 taskset -c 8-13 $E -I audit_mix.py --mix "$OUT" --base $BASE --carla "$CONV" --canonical-frames $CANON \
  --expected-canonical-sha 13abb7b25a0da8651221734bc1adf431e630c6006bc1027d215cfb5090778a45 --out "$OUT.audit.json" | tail -30 || true
sha256sum $CANON
printf '{"dataset_path": "%s"}\n' "$OUT" > "$OUT.verify_ready.json"
nice -n 10 ~/projects/detector-program/env310/bin/python "/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/detector_program/verify_second_mix.py" "$OUT.verify_ready.json" $BASE && echo CODEX_VERIFY_PASSED
