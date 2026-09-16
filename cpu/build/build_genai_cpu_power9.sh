#!/bin/bash
# Build onnxruntime-genai 0.15.0 (CPU) for ppc64le / POWER9,
# linking against the locally built ONNX Runtime via ORT_HOME.
set -x

ENV=/root/miniforge3/envs/onnx_build
SRC=/root/onnx/genai
BUILD=/root/onnx/build/genai-cpu
ORT_HOME=/root/onnx/ort-home-cpu

export PATH="$ENV/bin:$PATH"
export CC="$ENV/bin/powerpc64le-conda-linux-gnu-gcc"
export CXX="$ENV/bin/powerpc64le-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$ENV/lib:$ORT_HOME/lib:$LD_LIBRARY_PATH"

cd "$SRC" || exit 1

python build.py \
  --config Release \
  --build_dir "$BUILD" \
  --ort_home "$ORT_HOME" \
  --parallel \
  --skip_tests \
  --no_telemetry \
  --cmake_extra_defines \
      CMAKE_C_COMPILER="$CC" \
      CMAKE_CXX_COMPILER="$CXX" \
      CMAKE_PREFIX_PATH="$ENV"

echo "=== EXIT CODE: $? ==="
find "$BUILD" -name "*.whl" -o -name "libonnxruntime-genai.so" 2>/dev/null
