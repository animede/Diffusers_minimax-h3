#!/bin/bash
set -euo pipefail
cd /home/animede/minimax-h3/third_party/SageAttention
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export NVCC_THREADS=2
echo "=== Build starting $(date) ==="
echo "CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
/home/animede/minimax-h3/venv/bin/python -c "from torch.utils.cpp_extension import CUDA_HOME as CH; print('torch sees CUDA_HOME=', CH)"
/home/animede/minimax-h3/venv/bin/pip wheel . --no-build-isolation --no-deps -w /home/animede/minimax-h3/third_party/wheels
echo "=== Build finished $(date), exit=$? ==="
