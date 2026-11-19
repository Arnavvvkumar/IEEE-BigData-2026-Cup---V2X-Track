#!/bin/bash
# Combine the two gains: second pedestrian model (V2, pedestrians only) + twin-trained SECOND in place of the weaker old-data seed.
# Then the static-map pedestrian down-weighting on the result.
set -uo pipefail
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
P="${BDC_PYTHON}"
cd "$here"
F="$here/fusion"
A="$F/canon_all_s1234_ep12_allpred.json"; B="$F/canon_relabel_s1234_ep12_allpred.json"
V="$F/vnx010_all_s1234_ep10_allpred.json"; V2="$F/vnx010_all_rs2_amp_ep12_allpred.json"; T="$F/second_twin_s1_ep12_allpred.json"
VT="vehicle,truck"; PV="pedestrian,vehicle"
echo "== V ped+veh, V2 ped, SECOND all+twin+relabel"; "$P" fuse.py "$F/fuse_cw_d.json" "$V::$PV" "$V2::pedestrian" "$A::$VT" "$T::$VT" "$B::$VT" | grep fused
echo "== V ped+veh, V2 ped+veh, SECOND all+twin+relabel"; "$P" fuse.py "$F/fuse_cw_e.json" "$V::$PV" "$V2::$PV" "$A::$VT" "$T::$VT" "$B::$VT" | grep fused
cd "$here/../twin"
for n in d e; do
  echo "== static filter on fuse_cw_$n"
  "$P" static_filter.py "$F/fuse_cw_${n}_predictions.json" "$here/../twin/static_filter_cw_$n.json" | grep -E 'base|ped_scale0.8|all_scale0.8_f0.3|best'
done
echo FUSE6_DONE
