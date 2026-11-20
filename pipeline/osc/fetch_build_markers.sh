#!/bin/bash
# Print env-build milestones only. Avoids du, which is slow while rsync is writing.
set -euo pipefail
ssh -i "$HOME/.ssh/osc_ed25519" -o BatchMode=yes -o ConnectTimeout=40 ${OSC_USER}@ascend.osc.edu \
  "grep -E 'BUILD_START|ENV_DONE|BUILD_DONE|SUBMIT_DONE|ERROR|Successfully installed torch|Successfully installed spconv|Successfully installed numpy|GPU_ARGS|ModuleNotFound|gcc |python_mod|nvcc |sinfo|partition' /fs/scratch/${OSC_PROJECT}/bdc/logs/build_env.log | tail -n 80" \
  2>/dev/null | grep -v -E '^\*|authorized users|this computer|authority|subject to|and recorded|activities of|expressly|improperly|monitoring|personnel|officials|^$'
