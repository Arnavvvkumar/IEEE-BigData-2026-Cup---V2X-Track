#!/bin/bash
# CPU: convert CARLA v3, check it, and build two ground-canonical training sets.
set -euo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
cd "$here"
[ -d "$D/claude_carla_v3" ] || "$P" convert_carla_v3.py --native native_v3_Town02 native_v3_Town01 native_v3_Town03 native_v3_Town05 --out "$D/claude_carla_v3"
"$P" tools/check_v3.py "$D/claude_carla_v3" | cut -c1-330
R="$D/claude_v2carla_relabel_mix6000_v1"
# set A: CARLA only (1k v1 + 4.8k v3)
"$P" canon/build_premix.py --relabelled "$R" --take carla --v3 "$D/claude_carla_v3" --out "$D/claude_premix_carlaonly_v1"
"$P" canon/build_canon.py --src "$D/claude_premix_carlaonly_v1" --out "$D/claude_canon_carlaonly_v1" --ground -3.0 --workers 10 | tail -22
# set B: everything (5k V2 + 1k v1 + 4.8k v3)
"$P" canon/build_premix.py --relabelled "$R" --take both --v3 "$D/claude_carla_v3" --out "$D/claude_premix_all_v1"
"$P" canon/build_canon.py --src "$D/claude_premix_all_v1" --out "$D/claude_canon_all_v1" --ground -3.0 --workers 10 | tail -22
cd canon
"$P" make_cfg.py second_canon_v1.yaml "$D/claude_canon_carlaonly_v1" second_canon_carlaonly_v1.yaml
"$P" make_cfg.py second_canon_v1.yaml "$D/claude_canon_all_v1" second_canon_all_v1.yaml
echo SETS_DONE
