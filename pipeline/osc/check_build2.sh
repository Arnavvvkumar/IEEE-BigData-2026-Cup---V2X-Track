#!/bin/bash
ssh -i ~/.ssh/osc_ed25519 -o BatchMode=yes <osc_user>@ascend.osc.edu 'bash -s' 2>/dev/null << 'EOF' | cut -c1-180
R=/fs/scratch/<osc_project>/bdc
grep -E 'BUILD_DONE|iou3d_nms_cuda|^pcdet|error:|SUBMIT_DONE|Submitted|GPU_ARGS|Traceback' $R/logs/build_env_run2.out | tail -n 8
squeue -u <osc_user> | tail -n 3
ls $R/logs | grep smoke | tail -n 3
EOF
