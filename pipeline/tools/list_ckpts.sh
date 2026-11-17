#!/bin/bash
cd ~/projects/detector-program/OpenPCDet/output
find . -name 'checkpoint_epoch_*.pth' | sed 's#/ckpt/checkpoint_epoch_# #; s#\.pth##' | sort -k1,1 -k2,2n | awk '{a[$1]=a[$1]" "$2} END{for(k in a)print k":"a[k]}'
df -h ~ | tail -1
