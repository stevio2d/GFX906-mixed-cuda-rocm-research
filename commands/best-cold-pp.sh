#!/usr/bin/env bash
set -euo pipefail

# Highest confirmed cold PP command from this research cycle.

BIN_PATH="${BIN_PATH:-/home/stefan/llama.cpp-gfx906/build-cuda-hip-dl-gfx906/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/home/stefan/.lmstudio/models/unsloth/MiniMax-M2.7-GGUF/MiniMax-M2.7-UD-Q4_K_S-00001-of-00004.gguf}"
PORT="${PORT:-21523}"

export GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS="${GGML_CUDA_FATTN_TILE_PARALLEL_BACKENDS:-hip}"
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN="${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MIN:-321}"
export GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX="${GGML_CUDA_FATTN_TILE_PARALLEL_Q_MAX:-321}"
export GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS="${GGML_CUDA_FATTN_TILE_PARALLEL_BLOCKS:-1}"

exec "$BIN_PATH" \
  -m "$MODEL_PATH" \
  -c 69376 -ngl 999 -sm layer \
  -dev ROCm1,ROCm0,CUDA3,CUDA1,CUDA0,CUDA2 \
  -fa on -np 1 -fit off --numa distribute --no-warmup --jinja \
  -t 24 -tb 48 -lv 3 \
  -ctk q4_0 -ctv q4_0 \
  -b 800 -ub 800 \
  --placement-policy memory \
  --device-speed 0.55,0.55,1.0,1.0,1.0,1.0 \
  --cuda-fattn-no-stream-k \
  --cuda-fattn-vec-qkv-headroom-mib 0 \
  --reserve-pp-outputs 1 \
  --layer-split 14,12,7,11,11,7 \
  --output-device CUDA3 \
  --cuda-fattn-mma-q4-0-direct \
  --cuda-fattn-mma-q4-0-direct-mode all \
  --host 127.0.0.1 \
  --port "$PORT"
