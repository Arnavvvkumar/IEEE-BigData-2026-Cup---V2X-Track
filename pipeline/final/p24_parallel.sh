#!/bin/bash
# P24: maximum parallelism for P2 + V1 scoring (replaces P23 and P22's remaining steps).
#  - V1 inference starts now, concurrently with P2's inference (GPU has memory headroom).
#  - When P2's inference ends, stop P22 (its scoring is sequential) and run P2's 4 scoring settings in parallel.
#  - V1's 4 scoring settings run in parallel once V1 inference ends.
set -uo pipefail
base="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight"
F="$base/final"
P="$HOME/projects/detector-program/env310/bin/python"
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill "$1" 2>/dev/null; }
V10="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck second_twin_ll_s1_ep12:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
echo "P24_START $(date -Is)"
# 1) let P23's fetch finish, then stop P23 and the old watcher
until grep -q FETCHED "$F/p23_V1fast.log"; do sleep 5; done
for p in $(pgrep -f "^bash p23_V1fast.sh") $(pgrep -f "p22_stop_after_P2.sh"); do kill_tree "$p"; done
echo "V1 fetched; P23 + watcher stopped $(date -Is)"
# 2) V1 inference now, in the background
( BATCH=1 bash "$F/run_infer_one.sh" pvrcnn_veh_V1 pvrcnn_veh.yaml claude_canon_all_twin_st_v1 5 > "$F/p24_V1_infer.log" 2>&1; echo "V1 inference done $(date -Is)" ) &
vinf=$!
# 3) P2: wait for its inference to finish, stop P22, score 4 settings in parallel
while pgrep -f "run_infer_one.sh pvrcnn_ped_P2" > /dev/null; do sleep 10; done
for p in $(pgrep -f "^bash p22_P2V1.sh"); do kill_tree "$p"; done
echo "P2 inference done; P22 stopped $(date -Is)"
cd "$base/coop"
for s in dev150 dev350; do for w in 1 3; do
  (x=$(for i in $(seq $w); do printf '%s ' "pvrcnn_ped_P2_ep5:pedestrian"; done)
   "$P" eval150.py $s pvrcnn_ped_P2w${w}_norule $V10 $x norule > /dev/null
   "$P" verify_hedge_fn.py $s pvrcnn_ped_P2w${w}_norule | tail -n 1 | sed "s/^/$s P2 x$w: /") &
done; done
# 4) V1 scoring after its inference
wait $vinf
for s in dev150 dev350; do for w in 1 3; do
  (x=$(for i in $(seq $w); do printf '%s ' "pvrcnn_veh_V1_ep5:vehicle"; done)
   "$P" eval150.py $s pvrcnn_veh_V1w${w}_norule $V10 $x norule > /dev/null
   "$P" rule_retune.py $s pvrcnn_veh_V1w${w}_norule 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$s + V1 x$w: /") &
done; done
wait
echo "P24_DONE $(date -Is)"
