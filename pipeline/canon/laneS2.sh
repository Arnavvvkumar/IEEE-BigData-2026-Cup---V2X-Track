#!/bin/bash
# Lane S2: second seed of the long-low-truck SECOND, started in parallel with second_twin_ll_s1 (the GPU is only ~30% used by one run:
# the single training process is CPU-bound in Python, not data- or GPU-bound). Not seeded, so a different random seed than s1.
here="${BDC_ROOT}/experiments/claude_carla_overnight/canon"
D="${BDC_DATA}"
cd "$here"
[ -f "$D/claude_canon_all_twin_ll_v1/custom_infos_train.pkl" ] || { echo NO_DATA; exit 1; }
export WORKERS=2
bash "$here/run_arm_par_ns.sh" second_twin_ll_s2 "$here/second_canon_all_v1.yaml" "$D/claude_canon_all_twin_ll_v1" 12 '-3.15 -3.0 -2.85 -2.7' --use_amp
echo LANES2_DONE
