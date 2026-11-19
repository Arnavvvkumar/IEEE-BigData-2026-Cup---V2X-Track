#!/bin/bash
# Second fine-grid VoxelNeXt seed (old data): dump at six heights, then class-wise fusion with two pedestrian models.
set -uo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
P="${BDC_PYTHON}"
G='-3.15 -3.075 -3.0 -2.925 -2.85 -2.775'
cd "$here"
F="$here/fusion"
[ -f "$F/vnx010_all_rs2_amp_ep12_allpred.json" ] || bash dump_arm.sh vnx010_all_rs2_amp "$here/vnx010_all.yaml" "$D/claude_canon_all_v1" 12 "$G"
while pgrep -f 'bash run_fuse4.sh' > /dev/null; do sleep 20; done
A="$F/canon_all_s1234_ep12_allpred.json"; B="$F/canon_relabel_s1234_ep12_allpred.json"; R2="$F/canon_all_rs2_ep12_allpred.json"
V="$F/vnx010_all_s1234_ep10_allpred.json"; V2="$F/vnx010_all_rs2_amp_ep12_allpred.json"; T="$F/second_twin_s1_ep12_allpred.json"
VT="vehicle,truck"; PV="pedestrian,vehicle"
echo "== two VNX (ped+veh) + SECOND x3"; "$P" fuse.py "$F/fuse_cw_v2.json" "$V::$PV" "$V2::$PV" "$A::$VT" "$R2::$VT" "$B::$VT" | grep fused
echo "== two VNX (V ped+veh, V2 ped only) + SECOND x3"; "$P" fuse.py "$F/fuse_cw_v2p.json" "$V::$PV" "$V2::pedestrian" "$A::$VT" "$R2::$VT" "$B::$VT" | grep fused
echo "== two VNX (ped+veh) + SECOND x3 + twin SECOND"; "$P" fuse.py "$F/fuse_cw_v2T.json" "$V::$PV" "$V2::$PV" "$A::$VT" "$R2::$VT" "$B::$VT" "$T::$VT" | grep fused
echo FUSE5_DONE
