#!/bin/bash
# Read-only: sizes and versions needed to rebuild the training environment on a cluster.
D=~/projects/bdc2026-track-b/data
for d in claude_canon_all_v1 claude_twin_v1 claude_canon_all_twin_v1 claude_canon_relabel_mix6000_v1; do echo -n "$d "; du -shL "$D/$d" | cut -f1; done
echo -n 'OpenPCDet(no output) '; du -sh --exclude=output ~/projects/detector-program/OpenPCDet | cut -f1
echo -n 'OpenPCDet/output (checkpoints) '; du -sh ~/projects/detector-program/OpenPCDet/output | cut -f1
echo -n 'env310 '; du -sh ~/projects/detector-program/env310 | cut -f1
~/projects/detector-program/env310/bin/python -c "import torch,sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"
~/projects/detector-program/env310/bin/pip list 2>/dev/null | grep -i -E 'spconv|cumm|numpy|scipy|numba|pyyaml|easydict|torch'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc
