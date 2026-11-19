#!/bin/bash
# P27 (laptop; OSC read-only): P3 (8577952) and V2 (8577953) are finished. Fetch epoch 5, infer dev (batch 1),
# gate on top of the v11 list (B + P2 x3): P3 x1/x3 pedestrian, V2 x1 vehicle. Then per-scan test export (batch 1, fixed seed).
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P1_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian pvrcnn_ped_P2_ep5:pedestrian"
P3="pvrcnn_ped_P3_ep5:pedestrian"
V2="pvrcnn_veh_V2_ep5:vehicle"
export FUSE_YAW_CIRC=vehicle FUSE_ONE_VOTE=vehicle
echo "P27_START $(date -Is)"
for t in pvrcnn_ped_P3:pvrcnn_ped.yaml pvrcnn_veh_V2:pvrcnn_veh.yaml; do
  tag=${t%%:*}; cfg=${t#*:}
  bash "$base/osc/fetch_pv_ckpt.sh" 5 "$tag" | tail -n 1
  BATCH=1 bash "$base/final/run_infer_one.sh" "$tag" "$cfg" claude_canon_all_twin_st_v1 5 | tail -n 3
  echo "$tag dev inference done $(date -Is)"
done
cd "$base/coop"
score() {  # set name passes
  "$P" eval150.py $1 $2 $3 norule > /dev/null
  "$P" verify_hedge_fn.py $1 $2 | tail -n 1 | sed "s/^/$1 $2 PED: /"
  "$P" rule_retune.py $1 $2 2 'a 0.5 base 0.1' | grep 'a 0.5' | sed "s/^/$1 $2 VT: /"
}
for s in dev150 dev350; do
  score $s v12base_norule "$B" &
  score $s v12p3x1_norule "$B $P3" &
  wait
  score $s v12p3x3_norule "$B $P3 $P3 $P3" &
  score $s v12v2_norule "$B $V2" &
  wait
done
echo "P27_GATE_DONE $(date -Is)"
# per-scan test export for both (reexport script skips voters already exported)
sed -e 's/^run pvrcnn_veh_V1 .*/&\nrun pvrcnn_ped_P3        pvrcnn_ped.yaml           claude_canon_all_twin_st_v1       5 1\nrun pvrcnn_veh_V2        pvrcnn_veh.yaml           claude_canon_all_twin_st_v1       5 1/' \
  "$base/final/reexport_perscan_b1.sh" > "$base/final/_reexport_p27.sh"
bash "$base/final/_reexport_p27.sh" | grep -v '^skip'
echo "P27_DONE $(date -Is)"
