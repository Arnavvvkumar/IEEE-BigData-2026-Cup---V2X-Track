#!/bin/bash
#SBATCH --job-name=bdc_smoke
# Body script. Account, partition, and GPU are passed on the sbatch command line
# so the working GPU flag can be chosen after a --test-only probe.
set -euo pipefail
if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
module load python/3.10 cuda/11.8.0
source /fs/scratch/<osc_project>/bdc/env/bin/activate
cd /fs/scratch/<osc_project>/bdc/code/OpenPCDet
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "SMOKE_HOST $(hostname) $(date -Is)"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-unset}"
python -u /fs/scratch/<osc_project>/bdc/cfg/smoke_check.py
echo "SMOKE_JOB_DONE $(date -Is)"
