#!/bin/bash
# Read-only: sizes and versions needed to rebuild the training environment on a cluster.
D=${BDC_DATA}
for d in claude_canon_all_v1 claude_twin_v1 claude_canon_all_twin_v1 claude_canon_relabel_mix6000_v1; do echo -n "$d "; du -shL "$D/$d" | cut -f1; done
echo -n 'OpenPCDet(no output) '; du -sh --exclude=output ${BDC_PROGRAM}/OpenPCDet | cut -f1
echo -n 'OpenPCDet/output (checkpoints) '; du -sh ${BDC_PROGRAM}/OpenPCDet/output | cut -f1
echo -n 'env310 '; du -sh ${BDC_PROGRAM}/env310 | cut -f1
${BDC_PYTHON} -c "import torch,sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"
${BDC_PROGRAM}/env310/bin/pip list 2>/dev/null | grep -i -E 'spconv|cumm|numpy|scipy|numba|pyyaml|easydict|torch'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nproc
