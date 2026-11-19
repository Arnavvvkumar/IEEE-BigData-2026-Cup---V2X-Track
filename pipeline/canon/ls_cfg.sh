#!/bin/bash
# List detector configs / lane scripts and the size-rule grid lines so far.
cd "${BDC_ROOT}/experiments/claude_carla_overnight"
ls canon/*.yaml
ls canon/lane*.sh | tr '\n' ' '; echo
cat canon/run_arm_par_ns.sh
grep -E '^grow|base' coop/run_size_rule.log
ls ${BDC_PROGRAM}/OpenPCDet/tools/cfgs/kitti_models/ | tr '\n' ' '
