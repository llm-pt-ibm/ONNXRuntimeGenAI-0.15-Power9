#!/bin/bash
# Build onnxruntime-genai 0.15.0 com CUDA para ppc64le / POWER9 / Tesla V100 (sm_70),
# linkando contra o ORT CUDA construido localmente via ORT_HOME.
#
# Mesmo toolchain do ORT (conda GCC 13.3) -- o nvcc 12.4 aceita ate GCC 13.
set -x

ENV=/root/miniforge3/envs/onnx_build
SRC=/root/onnx/genai
BUILD=/root/onnx/build/genai-cuda
ORT_HOME=/root/onnx/ort-home-cuda
CUDA=/usr/local/cuda-12.4

export PATH="$ENV/bin:$CUDA/bin:$PATH"
export CC="$ENV/bin/powerpc64le-conda-linux-gnu-gcc"
export CXX="$ENV/bin/powerpc64le-conda-linux-gnu-g++"
export CUDA_HOME="$CUDA"
export LD_LIBRARY_PATH="$ENV/lib:$ORT_HOME/lib:$CUDA/lib64:$LD_LIBRARY_PATH"

cd "$SRC" || exit 1

python build.py \
  --config Release \
  --build_dir "$BUILD" \
  --ort_home "$ORT_HOME" \
  --use_cuda \
  --cuda_home "$CUDA" \
  --parallel \
  --skip_tests \
  --no_telemetry \
  --cmake_extra_defines \
      CMAKE_C_COMPILER="$CC" \
      CMAKE_CXX_COMPILER="$CXX" \
      CMAKE_CUDA_HOST_COMPILER="$CXX" \
      CMAKE_CUDA_ARCHITECTURES=70 \
      CMAKE_PREFIX_PATH="$ENV"

echo "=== EXIT CODE: $? ==="
find "$BUILD" -name "*.whl" -o -name "libonnxruntime-genai.so" 2>/dev/null
