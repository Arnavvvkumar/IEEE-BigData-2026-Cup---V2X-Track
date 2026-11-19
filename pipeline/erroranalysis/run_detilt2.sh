#!/bin/bash
# usage: run_detilt2.sh <name> <run key> <epoch> <plane mode> <ground z ...>
set -euo pipefail
root="${BDC_ROOT}"
program="${BDC_PROGRAM}"
repo="$program/OpenPCDet"; o="$repo/output"; data="${BDC_DATA}"
dp="$root/experiments/detector_program"; bx="$root/experiments/claude_carla_overnight/boxsize"
name=$1; key=$2; ep=$3; mode=$4; shift 4
case $key in
  mix) cfg="$dp/second_control_v1.yaml"; ds="$data/claude_v2carla_mix6000_v1"; ck="$o${BDC_ROOT}/experiments/detector_program/second_control_v1/overnight_mix_seed1234/ckpt";;
  control) cfg="$dp/second_control_v1.yaml"; ds="$data/codex_openpcdet_custom_v1"; ck="$o${BDC_ROOT}/experiments/detector_program/second_control_v1/overnight_control_seed1234/ckpt";;
  relabel) cfg="$bx/second_relabel_v1.yaml"; ds="$data/claude_v2carla_relabel_mix6000_v1"; ck="$o${BDC_ROOT}/experiments/claude_carla_overnight/boxsize/second_relabel_v1/relabel_containers_seed1234/ckpt";;
  pvrcnn) cfg="$bx/pvrcnn_relabel_b1_v1.yaml"; ds="$data/claude_v2carla_relabel_mix6000_v1"; ck="$o${BDC_ROOT}/experiments/claude_carla_overnight/boxsize/pvrcnn_relabel_b1_v1/pvrcnn_relabel_seed1234/ckpt";;
  vnx_b8) cfg="$repo/tools/cfgs/custom_models/voxelnext_v1.yaml"; ds="$data/codex_openpcdet_custom_v1"; ck="$o/custom_models/voxelnext_v1/codex_batch8_from_epoch6/ckpt";;
  vnx_xy010) cfg="$repo/tools/cfgs/custom_models/voxelnext_xy010_v1.yaml"; ds="$data/codex_openpcdet_custom_v1"; ck="$o/custom_models/voxelnext_xy010_v1/codex_xy010_full_seed1234/ckpt";;
  *) cfg=$CFG; ds=$DS; ck=$CK;;
esac
cd "$repo/tools"
"$program/env310/bin/python" "$root/experiments/claude_carla_overnight/erroranalysis/detilt_eval.py" --repo "$repo" --cfg "$cfg" --data "$ds" \
  --checkpoint "$ck/checkpoint_epoch_${ep}.pth" --out "$root/experiments/claude_carla_overnight/erroranalysis/detilt_$name.json" \
  --plane "$mode" --ground "$@" --save-best 2>&1 | grep -E '^(point-fit|median|points|none)|Error|Traceback'
