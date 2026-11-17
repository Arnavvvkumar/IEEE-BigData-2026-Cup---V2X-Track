#!/bin/bash
#SBATCH --job-name=bdc_pvrcnn
# OpenPCDet training on synthetic data only, from scratch: no real labels, no real object points, no pretrained weights.
# Account / partition / GPU / time come from sbatch flags.
# env: TAG (run name), EPOCHS, BS, WORKERS, CFG (config file name in cfg/, default pvrcnn_all.yaml),
#      DATA (dataset folder in data/, default claude_canon_all_twin_st_v1), LIMIT (optional seconds: stop early, test jobs only)
set -uo pipefail
if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
module load python/3.10 cuda/11.8.0
R=/fs/scratch/<osc_project>/bdc
source $R/env/bin/activate
cd $R/code/OpenPCDet/tools
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CFG=${CFG:-pvrcnn_all.yaml}; DATA=${DATA:-claude_canon_all_twin_st_v1}
echo "HOST $(hostname) $(date -Is) TAG ${TAG} CFG ${CFG} DATA ${DATA} EPOCHS ${EPOCHS} BS ${BS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cmd=(python -u train.py --cfg_file $R/cfg/$CFG --batch_size "${BS}" --epochs "${EPOCHS}" --workers "${WORKERS:-8}"
     --extra_tag "${TAG}" --skip_builtin_eval --logger_iter_interval 50 --wo_gpu_stat --max_ckpt_save_num 6
     --set DATA_CONFIG.DATA_PATH $R/data/$DATA)
if [ -n "${LIMIT:-}" ]; then
  timeout "${LIMIT}" "${cmd[@]}"; rc=$?
  echo "TEST_RC $rc (124 = stopped by the time limit, which is the expected outcome)"
else
  "${cmd[@]}"; rc=$?
  echo "TRAIN_RC $rc"
fi
echo "JOB_END $(date -Is)"
