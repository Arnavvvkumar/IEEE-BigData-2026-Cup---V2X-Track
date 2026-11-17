#!/bin/bash
# Build the Track B training env on ASCEND scratch, then compile OpenPCDet for A100 (sm_80).
# Runs on OSC. Uses OSC's outbound network only. Does not touch the laptop data upload.
set -euo pipefail
ROOT=/fs/scratch/<osc_project>/bdc
LOG=$ROOT/logs/build_env.log
mkdir -p "$ROOT/logs" "$ROOT/tmp" "$ROOT/.pip-cache" "$ROOT/out"
if [ -f "$ROOT/logs/build_env.pid" ]; then
  old=$(cat "$ROOT/logs/build_env.pid" || true)
  if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
    echo "build already running pid $old"
    exit 0
  fi
fi
echo $$ > "$ROOT/logs/build_env.pid"
exec > >(tee -a "$LOG") 2>&1
echo "BUILD_START $(date -Is) pid $$"
echo "host $(hostname)"

if ! type module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile
fi
module load python/3.10 cuda/11.8.0
module list
echo "python_mod $(command -v python) $(python --version 2>&1)"
echo "nvcc $(command -v nvcc)"
nvcc --version
echo "gcc $(gcc --version | head -n 1)"
echo "CUDA_HOME_before ${CUDA_HOME:-unset}"
if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
fi
echo "CUDA_HOME $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export PIP_CACHE_DIR=$ROOT/.pip-cache
export TMPDIR=$ROOT/tmp
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== partition ==="
sinfo -p debug-nextgen -o '%P %a %l %D %G' || true

if [ ! -x "$ROOT/env/bin/python" ]; then
  python -m venv "$ROOT/env"
fi
# shellcheck disable=SC1091
source "$ROOT/env/bin/activate"
echo "venv $(python -c 'import sys; print(sys.prefix)')"
if python - << 'PY'
import numpy, torch, spconv, cumm
assert numpy.__version__.startswith("1.23"), numpy.__version__
assert torch.__version__.startswith("2.2.2"), torch.__version__
print("ENV_ALREADY", numpy.__version__, torch.__version__, getattr(spconv, "__version__", "ok"))
PY
then
  echo "skipping pip; env already imports"
else
  python -m pip install -U pip wheel 'setuptools<70'
  python -m pip install -c "$ROOT/cfg/constraints.txt" numpy==1.23.5
  python -m pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --extra-index-url https://pypi.org/simple
  if ! python -m pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html; then
    python -m pip install torch_scatter==2.1.2+pt22cu118 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
  fi
  python -m pip install spconv-cu118==2.3.6 cumm-cu118==0.4.11 ninja==1.13.2
  if ! python -m pip install -c "$ROOT/cfg/constraints.txt" -r "$ROOT/cfg/requirements_osc.txt"; then
    echo "FULL_REQ_FAILED trying OpenPCDet minimal set"
    python -m pip install -c "$ROOT/cfg/constraints.txt" \
      llvmlite==0.41.1 numba==0.58.1 tensorboardX==2.6.2.2 easydict==1.13 \
      PyYAML==6.0.2 scikit-image==0.21.0 tqdm==4.67.1 SharedArray==3.2.4 \
      opencv-python-headless==4.8.1.78 pyquaternion==0.9.9 scipy==1.10.1
  fi
fi
python - << 'PY'
import numpy, torch
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda_build", torch.version.cuda)
assert numpy.__version__.startswith("1.23"), numpy.__version__
assert torch.__version__.startswith("2.2.2"), torch.__version__
import spconv, cumm
print("spconv_import_ok", getattr(spconv, "__version__", "unknown"))
PY
echo "ENV_DONE $(date -Is)"

echo "=== compile OpenPCDet sm_80 ==="
cd "$ROOT/code/OpenPCDet"
# The Intel icpc picked up by default breaks nvcc/torch headers; force GNU compilers (system gcc 11 matches CUDA 11.8's build).
module unload intel intel-oneapi-compilers 2>/dev/null || true
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
echo "host compiler: $($CXX --version | head -n 1)"
rm -rf build
export TORCH_CUDA_ARCH_LIST=8.0
export FORCE_CUDA=1
export MAX_JOBS=4
# Current setuptools turns `setup.py develop` into `pip install -e .` inside an
# isolated build env, which cannot import the torch just installed in this venv.
# Compile with this interpreter, then register the tree as editable.
python setup.py build_ext --inplace
python -m pip install -e . --no-build-isolation --no-deps
python -m pip install -c "$ROOT/cfg/constraints.txt" --force-reinstall --no-deps numpy==1.23.5
python - << 'PY'
import numpy, torch, pcdet
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
print("numpy", numpy.__version__)
print("pcdet", pcdet.__file__)
print("iou3d_nms_cuda", iou3d_nms_cuda.__file__)
assert numpy.__version__.startswith("1.23")
PY
echo "BUILD_DONE $(date -Is)"

echo "=== submit debug-nextgen smoke ==="
submit_with() {
  # $1 is extra sbatch gpu arguments, unquoted words
  # shellcheck disable=SC2086
  sbatch --test-only --account=<osc_project> --partition=debug-nextgen --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=32G --time=00:20:00 \
    --job-name=bdc_smoke \
    --output=$ROOT/logs/smoke_%j.out --error=$ROOT/logs/smoke_%j.err \
    $1 --wrap "true"
}

GPU_ARGS=""
if submit_with "--gpus-per-node=1"; then
  GPU_ARGS="--gpus-per-node=1"
elif submit_with "--gpus=1"; then
  GPU_ARGS="--gpus=1"
elif submit_with "--gres=gpu:1"; then
  GPU_ARGS="--gres=gpu:1"
else
  echo "SBATCH_GPU_FLAG_FAILED"
  exit 1
fi
echo "GPU_ARGS $GPU_ARGS"
# shellcheck disable=SC2086
sbatch --account=<osc_project> --partition=debug-nextgen --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=32G --time=00:20:00 \
  --job-name=bdc_smoke \
  --output=$ROOT/logs/smoke_%j.out --error=$ROOT/logs/smoke_%j.err \
  $GPU_ARGS \
  "$ROOT/cfg/smoke_debug.sbatch"
echo "SUBMIT_DONE $(date -Is)"
squeue -u <osc_user> || true
