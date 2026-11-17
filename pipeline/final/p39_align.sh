#!/bin/bash
# P39 (Codex's idea): equivalent-box alignment of pedestrian cluster members before averaging, on the v16 base. Audit (how many members are
# in the swapped description) + pedestrian AP (hedged), dev150 + dev350 + halves. Starts when the hedge re-tune has finished (RAM).
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
P="$HOME/projects/detector-program/env310/bin/python"
while pgrep -f "ped_retune.py" > /dev/null; do sleep 60; done
"$P" - <<'EOF'
import sys, numpy as np
sys.path.insert(0, '/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/final')
import fuse_test as F
g, k = F.align(np.array([[0, 0, 0, 1.0, 0.6, 1.7, 0.0], [0, 0, 0, 0.6, 1.0, 1.7, np.pi / 2], [0, 0, 0, 0.6, 1.0, 1.7, -np.pi / 2 + 0.1]]))
print('align unit test:', np.round(g, 3).tolist(), 'rewritten', k)
EOF
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
cd "$base/coop"
D="FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle"
echo "P39_START $(date -Is)"
for s in dev150 dev350; do
  env $D FUSE_ALIGN=pedestrian FUSE_AUDIT="$base/final/p39_audit_$s.json" "$P" eval150.py $s pa_align_norule $B norule > /dev/null
  echo "$s audit $(cat "$base/final/p39_audit_$s.json")"
  "$P" verify_hedge_fn.py $s pa_align_norule | tail -n 1 | sed "s/^/$s pa_align_norule PED: /"
done
echo "P39_DONE $(date -Is)"
