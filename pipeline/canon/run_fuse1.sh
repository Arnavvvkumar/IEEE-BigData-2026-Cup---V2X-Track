#!/bin/bash
# Dump multi-height predictions for the two best SECOND models, then score height fusion and model fusion.
set -uo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
cd "$here"
bash dump_arm.sh canon_all_s1234 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_v1" 12 "$G"
bash dump_arm.sh canon_relabel_s1234 "$here/second_canon_v1.yaml" "$D/claude_canon_relabel_mix6000_v1" 12 "$G"
F="$here/fusion"
echo "== heights, canon_all"; "$P" fuse.py "$F/fuse_all_heights.json" "$F/canon_all_s1234_ep12_allpred.json" | grep fused
echo "== heights 3 central, canon_all"; "$P" fuse.py "$F/fuse_all_h3.json" "$F/canon_all_s1234_ep12_allpred.json:-3.075,-3.0,-2.925,-2.85" | grep fused
echo "== heights, canon_relabel"; "$P" fuse.py "$F/fuse_relabel_heights.json" "$F/canon_relabel_s1234_ep12_allpred.json" | grep fused
echo "== two models, all heights"; "$P" fuse.py "$F/fuse_2model_heights.json" "$F/canon_all_s1234_ep12_allpred.json" "$F/canon_relabel_s1234_ep12_allpred.json" | grep fused
echo "== two models at -2.85"; "$P" fuse.py "$F/fuse_2model_285.json" "$F/canon_all_s1234_ep12_allpred.json:-2.85" "$F/canon_relabel_s1234_ep12_allpred.json:-2.85" | grep fused
echo FUSE1_DONE
