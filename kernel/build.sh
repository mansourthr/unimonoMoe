#!/usr/bin/env bash
# Compile one kernel variant into <variant>/build/libprefill_mono.so.
#
#   ./build.sh qwen_tp
#   ./build.sh qwen_ep
#   ./build.sh deepseek_tp
#   NVCC=/usr/local/cuda-13.0/bin/nvcc ./build.sh deepseek_tp
#
# The geometry a variant is compiled for lives in its .cu as constexpr ints, not
# on this command line, so the binary is the record of what it can run. The vLLM
# side reads that geometry back off the .so through prefill_wgmma_* accessors and
# declines the kernel when it does not match the loaded weights.
#
# sm_90a, not sm_90: the kernel uses WGMMA and TMA, which are Hopper
# architecture-specific and are unavailable under the portable sm_90 target.
#
# pipefail matters here. nvcc is piped through tee so the warnings stay readable,
# and without it the exit status is tee's, so a compile error would return 0.
set -eu
set -o pipefail

V=${1:-}
D=$(cd "$(dirname "$0")" && pwd)
if [ -z "$V" ] || [ ! -d "$D/$V/src" ]; then
  echo "usage: $0 <qwen_tp|qwen_ep|deepseek_tp>" >&2
  exit 2
fi
NVCC=${NVCC:-nvcc}

mkdir -p "$D/$V/build"
# The sources reach the one shared copy of the grid barrier and the PTX helpers
# through "../../moe_monokernel/src/...", relative to their own directory, so no
# extra include path is needed for it.
"$NVCC" -O3 -DNDEBUG -std=c++17 \
  --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] \
  --expt-relaxed-constexpr -lineinfo -Xptxas -v \
  -shared -Xcompiler -fPIC \
  -I "$D/$V/src" \
  "$D/$V"/src/*.cu -o "$D/$V/build/libprefill_mono.so" -lcuda 2>&1 \
  | tee "$D/$V/build/build.log"

echo "built $D/$V/build/libprefill_mono.so"
