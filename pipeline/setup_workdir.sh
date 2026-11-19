#!/bin/bash
# Create the working-directory layout the scripts expect, using links into this repository.
# usage: BDC_ROOT=/path/to/workdir bash pipeline/setup_workdir.sh /path/to/starting_kit/track_b
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/pipeline/env.sh"
KIT="${1:?usage: setup_workdir.sh /path/to/starting_kit/track_b}"
mkdir -p "$BDC_ROOT/experiments" "$BDC_ROOT/starting_kit" "$BDC_DATA" "$BDC_PROGRAM" "$BDC_OUT"
ln -sfn "$REPO/pipeline" "$BDC_ROOT/experiments/claude_carla_overnight"
ln -sfn "$REPO/generator_v2/scripts" "$BDC_ROOT/scripts"
ln -sfn "$REPO/generator_v2/configs" "$BDC_ROOT/configs"
ln -sfn "$(cd "$KIT" && pwd)" "$BDC_ROOT/starting_kit/track_b"
echo "workdir ready: $BDC_ROOT"
echo "next: install OpenPCDet into $BDC_PROGRAM/OpenPCDet and run 'source pipeline/env.sh' in every new shell"
