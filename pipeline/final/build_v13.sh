#!/bin/bash
# v13 = build_v11.sh pipeline (per-scan exports, no long-low voter, ADD voters) but with the v9 realism scans (twin/final_synthetic_v3).
# usage: ADD="..." bash build_v13.sh <name>.  Nothing is submitted.
set -euo pipefail
name=$1
here="${BDC_ROOT}/experiments/claude_carla_overnight/final"
sed -e 's#^bash "\$g" 2>&1#sed -i "s/final_synthetic_v4/final_synthetic_v3/g" "$g"; grep -c final_synthetic_v3 "$g"; &#' \
    -e 's#SYN=final_synthetic_v4#SYN=final_synthetic_v3#' "$here/build_v11.sh" > "$here/_build_v13_wrap.sh"
bash "$here/_build_v13_wrap.sh" "$name"
