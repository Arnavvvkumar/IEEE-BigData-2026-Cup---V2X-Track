#!/bin/bash
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes ${OSC_USER}@ascend.osc.edu "OSC_USER=${OSC_USER} OSC_PROJECT=${OSC_PROJECT} bash -s" 2>/dev/null << 'EOF' | cut -c1-180
R=/fs/scratch/${OSC_PROJECT}/bdc
grep -E 'BUILD_DONE|iou3d_nms_cuda|^pcdet|error:|SUBMIT_DONE|Submitted|GPU_ARGS|Traceback' $R/logs/build_env_run2.out | tail -n 8
squeue -u ${OSC_USER} | tail -n 3
ls $R/logs | grep smoke | tail -n 3
EOF
