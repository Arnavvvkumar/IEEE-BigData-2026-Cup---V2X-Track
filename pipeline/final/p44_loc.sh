#!/bin/bash
# P44: fetch the 2 approved LOC checkpoints (read-only scp, epoch 12), infer on dev150/dev350/test scans, then test them as pedestrian voters.
# Variants (pre-registered): REF = current v17 ped ensemble; +L1; +L2; +L12 (both). v17 recipe, 3 decimals, hedged. Dev150 first; dev350 only if a variant wins on dev150 and its halves.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
repo="${BDC_PROGRAM}/OpenPCDet"
echo "P44_START $(date -Is)"
for n in 1 2; do
  tag=vnx_ped_loc_$n
  dst="$repo/output/loc/$tag/ckpt"; mkdir -p "$dst"
  [ -f "$dst/checkpoint_epoch_12.pth" ] || timeout 600 scp -o ServerAliveInterval=15 -o ServerAliveCountMax=4 $K "<osc_user>@ascend.osc.edu:/fs/scratch/<osc_project>/bdc/code/OpenPCDet/output/fs/scratch/<osc_project>/bdc/cfg/vnx010_ped_loc/$tag/ckpt/checkpoint_epoch_12.pth" "$dst/" 2>/dev/null
  ls -la "$dst/checkpoint_epoch_12.pth"
  BATCH=1 bash "$base/final/run_infer_one.sh" $tag vnx010_ped_loc.yaml claude_canon_all_v1 12
done
B="vnx010_all_s1234_ep10 vnx010_twin_s1_ep12 vnx010_all_rs2_amp_ep12 vnx010_twin_s1_tta_ep12 vnx010_all_s1234_tta_ep10 pvrcnn_st_osc_ep6 pvrcnn_st_osc_s2_ep4"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
BASEPED=""; for t in $B; do BASEPED="$BASEPED $t:pedestrian"; done
for t in pvrcnn_ped_P1_ep5 pvrcnn_ped_P2_ep5 pvrcnn_ped_P3_ep5 pvrcnn_ped_P1_tta_ep5 pvrcnn_ped_P3_tta_ep5; do BASEPED="$BASEPED $(x3 $t)"; done
BASEPED="$BASEPED pvrcnn_ped_P4_ep5:pedestrian"
NONPED="vnx010_pv_ft_ep5 vnx010_pv_ft2_ep5 canon_all_s1234_ep12 second_twin_s1_ep12 canon_relabel_s1234_ep12 second_twin_s2_ep12 second_twin_st_s1_ep12 second_twin_pv_s1_ep14 vnx010_pv_ft4_ep4"
NP=""; for t in $NONPED; do NP="$NP $t:none"; done
cd "$base/coop"
export FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle
S=${1:-dev150}
run() { local name=$1; shift; "$P" eval150.py $S $name $NP $BASEPED "$@" norule > /dev/null 2>&1; "$P" ped_ap3.py $S $name; }
echo "REF";  run loc_ref
echo "+L1";  run loc_L1 vnx_ped_loc_1_ep12:pedestrian
echo "+L2";  run loc_L2 vnx_ped_loc_2_ep12:pedestrian
echo "+L12"; run loc_L12 vnx_ped_loc_1_ep12:pedestrian vnx_ped_loc_2_ep12:pedestrian
echo "P44_DONE $S $(date -Is)"
