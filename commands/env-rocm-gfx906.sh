#!/usr/bin/env bash
set -euo pipefail

export PATH=/opt/rocm/bin:$PATH
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCBLAS_LAYER=0
export ROCBLAS_LOG_LEVEL=0
export HIPBLASLT_LOG_LEVEL=0
export HIP_FORCE_DEV_KERNARG=1
export GPU_MAX_HW_QUEUES=8
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-custom/lib/rocblas/library
export LD_LIBRARY_PATH=/opt/rocm-custom/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
