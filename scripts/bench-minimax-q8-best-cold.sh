#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/stefan/llama.cpp-gfx906}
BIN=${BIN:-$ROOT/build-cuda-hip-dl-gfx906/bin/llama-server}
MODEL=${MODEL:-/home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf}
PORT=${PORT:-21941}

export PATH=/opt/rocm/bin:${PATH:-}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-9.0.6}
export ROCBLAS_LAYER=${ROCBLAS_LAYER:-0}
export ROCBLAS_LOG_LEVEL=${ROCBLAS_LOG_LEVEL:-0}
export HIPBLASLT_LOG_LEVEL=${HIPBLASLT_LOG_LEVEL:-0}
export HIP_FORCE_DEV_KERNARG=${HIP_FORCE_DEV_KERNARG:-1}
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-8}
export ROCBLAS_TENSILE_LIBPATH=${ROCBLAS_TENSILE_LIBPATH:-/opt/rocm-custom/lib/rocblas/library}
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Exact benchmark prompt tuning for the known 11521-token cold payload family.
export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS=${GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS:-hip}
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN=${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN:-321}
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX=${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX:-321}
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS=${GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS:-1}

exec "$BIN" \
  -m "$MODEL" \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q8_0 -ctv q8_0 \
  -b 200 -ub 200 \
  --placement-policy balanced \
  --device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
  --cuda-fattn-no-stream-k \
  --cuda-fattn-vec-qkv-headroom-mib 0 \
  --reserve-pp-outputs 1 \
  --cuda-fattn-mma-q4-0-direct \
  --cuda-fattn-mma-q4-0-direct-mode all \
  --output-device CUDA3 \
  --cache-prompt \
  --cache-ram 32768 \
  --host 127.0.0.1 \
  --port "$PORT"
