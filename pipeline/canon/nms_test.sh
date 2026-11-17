#!/bin/bash
# Does the VoxelNeXt detection cap (class-agnostic NMS 0.2, max 83 boxes/frame, score>=0.05) cost pedestrian AP?
# Variants of the same checkpoint (vnx010_all_s1234 ep10); only post-processing changes. Evaluation on the real dev frames only.
set -uo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight/canon"
program="$HOME/projects/detector-program"
repo="$program/OpenPCDet"
D="$HOME/projects/bdc2026-track-b/data"
G='-3.075 -3.0 -2.925 -2.85'
mkdir -p "$here/nms"
mk() {  # name post thresh score maxobj
  sed -e "s/NMS_POST_MAXSIZE: 83/NMS_POST_MAXSIZE: $2/" -e "s/NMS_THRESH: 0.2/NMS_THRESH: $3/" -e "s/SCORE_THRESH: 0.05/SCORE_THRESH: $4/" \
      -e "s/MAX_OBJ_PER_SAMPLE: 500/MAX_OBJ_PER_SAMPLE: $5/" "$here/vnx010_all.yaml" > "$here/nms/$1.yaml"
}
mk base 83 0.2 0.05 500
mk cap500 500 0.2 0.01 1000
mk nms5 500 0.5 0.01 1000
ck=$(dirname "$(find "$repo/output" -path "*/vnx010_all_s1234/ckpt/checkpoint_epoch_*.pth" -print -quit)")
cd "$repo/tools"
for v in base cap500 nms5; do
  "$program/env310/bin/python" "$root/experiments/claude_carla_overnight/erroranalysis/detilt_eval.py" --repo "$repo" --cfg "$here/nms/$v.yaml" \
    --data "$D/claude_canon_all_v1" --checkpoint "$ck/checkpoint_epoch_10.pth" --out "$here/nms/$v.json" --plane median --ground $G --save-all 2>&1 \
    | grep -E '^median|Error|Traceback' | sed "s/^/$v /" | tee -a "$here/nms/summary.txt"
  sleep 20
done
echo NMS_TEST_DONE
