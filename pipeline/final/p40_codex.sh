#!/bin/bash
# P40: Codex's ideas on the v16 base, pedestrian AP at full precision (ped_ap2.py), dev150 + dev350 + halves. Pre-registered:
#  agreement ranking lambda 0.25 / 0.5; robust vertical fusion replace / 50-50; centroid refine radial-only / radial + 0.5 tangential.
# Fast path: vehicle/truck classes are dropped from every pass ('none'), which leaves pedestrian fusion, static down-weight, calibration
# and the pass count (score = sum/passes) unchanged. Checked first: base must reproduce the full-spec pedestrian AP.
set -uo pipefail
base="${BDC_ROOT}/experiments/claude_carla_overnight"
P="${BDC_PYTHON}"
B="vnx010_all_s1234_ep10:pedestrian,vehicle vnx010_twin_s1_ep12:pedestrian,vehicle vnx010_pv_ft_ep5:vehicle vnx010_pv_ft2_ep5:vehicle vnx010_all_rs2_amp_ep12:pedestrian vnx010_twin_s1_tta_ep12:pedestrian vnx010_all_s1234_tta_ep10:pedestrian canon_all_s1234_ep12:vehicle,truck second_twin_s1_ep12:vehicle,truck canon_relabel_s1234_ep12:vehicle,truck second_twin_s2_ep12:vehicle,truck second_twin_st_s1_ep12:truck second_twin_pv_s1_ep14:truck vnx010_pv_ft4_ep4:vehicle pvrcnn_st_osc_ep6:pedestrian,vehicle,truck pvrcnn_st_osc_s2_ep4:pedestrian,vehicle"
x3() { echo "$1:pedestrian $1:pedestrian $1:pedestrian"; }
B="$B $(x3 pvrcnn_ped_P1_ep5) $(x3 pvrcnn_ped_P2_ep5) $(x3 pvrcnn_ped_P3_ep5) $(x3 pvrcnn_ped_P1_tta_ep5) $(x3 pvrcnn_ped_P3_tta_ep5) pvrcnn_ped_P4_ep5:pedestrian"
PB=""; for s in $B; do t=${s%%:*}; c=${s##*:}; [[ ",$c," == *",pedestrian,"* ]] && PB="$PB $t:pedestrian" || PB="$PB $t:none"; done
cd "$base/coop"
D="FUSE_YAW_CIRC=vehicle,pedestrian FUSE_ONE_VOTE=vehicle"
run() { env $D $3 "$P" eval150.py $1 $2 $PB norule > /dev/null; "$P" ped_ap2.py $1 $2; }
echo "P40_START $(date -Is)"
"$P" ped_ap2.py dev150 pf_yawcirc_norule | sed 's/^/REFERENCE full spec: /'
run dev150 px_base_norule ""
for s in dev150 dev350; do
  [ $s = dev350 ] && { "$P" ped_ap2.py dev350 pf_yawcirc_norule | sed 's/^/REFERENCE full spec: /'; run dev350 px_base_norule ""; }
  run $s px_agree25_norule "FUSE_PED_AGREE=0.25" &
  run $s px_agree50_norule "FUSE_PED_AGREE=0.5" &
  run $s px_zmed100_norule "FUSE_PED_ZMED=1" &
  wait
  run $s px_zmed50_norule "FUSE_PED_ZMED=0.5" &
  run $s px_lat0_norule "CALIB_PED_LAT=0" &
  run $s px_lat05_norule "CALIB_PED_LAT=0.5" &
  wait
done
echo "P40_DONE $(date -Is)"
