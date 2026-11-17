#!/bin/bash
# CPU only: PV-RCNN scratch batch-1 config on the relabelled mix (anchors = relabelled training medians), then CPU preflight.
set -euo pipefail
root="/mnt/c/Users/arnav/Downloads/BIG DATA IEEE"
here="$root/experiments/claude_carla_overnight"
program="$HOME/projects/detector-program"
py="$program/env310/bin/python"
data="$HOME/projects/bdc2026-track-b/data/claude_v2carla_relabel_mix6000_v1"
cfg="$here/boxsize/pvrcnn_relabel_b1_v1.yaml"
cd "$here/pvrcnn"
[ -f "$cfg" ] || "$py" make_pvrcnn_config.py --repo "$program/OpenPCDet" --shared "$root/experiments/detector_program/second_control_v1.yaml" \
  --base "$data" --data-path "$data" --max-voxels 80000 --out "$cfg"
grep -A8 'class_name' "$cfg" | grep -E 'class_name|anchor_sizes|- -|bottom' -A1 | head -40
"$py" pvrcnn_preflight.py --cfg "$cfg" --data-path "$data" --out "$here/boxsize/preflight_pvrcnn_relabel.json" > /dev/null
"$py" -c "import json,sys;d=json.load(open(sys.argv[1]));print(json.dumps({k:v for k,v in d.items() if k!='frames'},indent=0)[:3500])" \
  "$here/boxsize/preflight_pvrcnn_relabel.json"
