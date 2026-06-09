#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

BUILD_DIR=${BUILD_DIR:-"$ROOT/build-cuda-hip-dl-gfx906-fast"}
JOBS=${JOBS:-12}

# Host-specific defaults:
# - RTX 3090: native sm_86
# - RTX 5070 Ti with CUDA 12.0: no native sm_120 support, so keep compute_90 PTX for driver JIT
# - MI50: gfx906
CUDA_ARCHS=${CUDA_ARCHS:-"86-real;90-virtual"}
GPU_TARGETS=${GPU_TARGETS:-"gfx906"}

export PATH="/opt/rocm/bin:${PATH:-}"
export LD_LIBRARY_PATH="/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH:-}"

CCACHE_ARGS=()
if command -v ccache >/dev/null 2>&1; then
    CCACHE_ARGS=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
        -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
    )
fi

cmake_args=(
    -S "$ROOT"
    -B "$BUILD_DIR"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
    -DGPU_TARGETS="$GPU_TARGETS"
    -DBUILD_SHARED_LIBS=ON
    -DGGML_BACKEND_DL=ON
    -DGGML_NATIVE=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_FA=ON
    -DGGML_CUDA_GRAPHS=OFF
    -DGGML_CUDA_NCCL=ON
    -DGGML_CUDA_COMPRESSION_MODE=size
    -DGGML_HIP=ON
    -DGGML_HIP_GRAPHS=OFF
    -DGGML_HIP_NO_VMM=ON
    -DGGML_HIP_MMQ_MFMA=ON
    -DGGML_BUILD_EXAMPLES=OFF
    -DGGML_BUILD_TESTS=OFF
    -DBUILD_TESTING=OFF
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_OPENSSL=OFF
    -DLLAMA_TESTS_INSTALL=OFF
    -DLLAMA_TOOLS_INSTALL=OFF
    "${CCACHE_ARGS[@]}"
)

cmake "${cmake_args[@]}"

cmake --build "$BUILD_DIR" --target llama-server --parallel "$JOBS"
