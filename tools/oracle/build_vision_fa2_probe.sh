#!/usr/bin/env bash
# Build only the production vision adapters for the exact-bit Python probe.
set -euo pipefail
export PATH="/opt/cuda/bin:$PATH" CUDA_VISIBLE_DEVICES=0
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
V="$ROOT/crates/w34-kernels/vendor/vision-fa2"
nvcc -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --extended-lambda \
  -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 \
  -gencode=arch=compute_89,code=compute_89 -Xcompiler=-fPIC -shared \
  -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI \
  -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_LOCAL \
  -include "$V/flash-attn/fa2_compat_prelude.h" \
  -I"$V/flash-attn/stubs" -I"$V/flash-attn/src" -I"$V/cutlass/include" \
  "$ROOT/crates/w34-kernels/csrc/ops/op_vision_fa2.cu" \
  "$ROOT/crates/w34-kernels/csrc/ops/op_vision.cu" \
  -o "${1:-/tmp/w34-vision-fa2.so}"
