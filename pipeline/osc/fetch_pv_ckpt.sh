#!/bin/bash
# Read-only on OSC: copy one PV-RCNN checkpoint of an approved job to the laptop, into the layout final/run_infer_one.sh expects.
# No job is submitted or changed.  usage: fetch_pv_ckpt.sh <epoch> [tag=pvrcnn_st_osc]
# The tag is matched as a whole path component (/<tag>/ckpt/), so pvrcnn_st_osc and pvrcnn_st_osc_s2 are never confused.
e=${1:?epoch}; tag=${2:-pvrcnn_st_osc}
K="-i $HOME/.ssh/osc_ed25519 -o BatchMode=yes -o ConnectTimeout=60"
remote=$(ssh $K <osc_user>@ascend.osc.edu "find /fs/scratch/<osc_project>/bdc/code/OpenPCDet/output -path '*/$tag/ckpt/checkpoint_epoch_${e}.pth'" 2>/dev/null | grep checkpoint_epoch | head -n 1)
[ -n "$remote" ] || { echo "NO_REMOTE_CKPT $tag epoch $e"; exit 1; }
dst="$HOME/projects/detector-program/OpenPCDet/output/mnt/c/Users/arnav/Downloads/BIG DATA IEEE/experiments/claude_carla_overnight/canon/pvrcnn_all/$tag/ckpt"
mkdir -p "$dst"
scp $K "<osc_user>@ascend.osc.edu:$remote" "$dst/" 2>/dev/null
ls -la "$dst" | tail -n 2
echo "FETCHED $tag epoch $e from $remote"
