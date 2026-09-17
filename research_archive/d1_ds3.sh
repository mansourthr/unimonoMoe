#!/bin/bash
# DeepSeek-V3 EP prefill / TTFT campaign, 8 x H200, TP=8 / DP=1 / PCP=1 / world 8.
#
# WHY A FRESH CAMPAIGN AND NOT A JOIN AGAINST c2camp. The existing DS3 arms cannot supply a stock
# Triton EP baseline:
#   c2camp `ep`  carries VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2, so it is ALREADY optimized.
#   c2camp `tri` has EP OFF, so it is a different parallel topology and confounds the comparison.
#   ep_all6 `gbase` carries VLLM_EP_CG_WORST=1 VLLM_EP_MASKED_SUM=1, and `ebase` / `cbase` are
#                   DECODE arms in ms/token, not prefill.
# An arm label is not its composition. So `dtri` below is built explicitly: EP ON, monokernel OFF,
# and NO VLLM_EP_* patch of any kind.
#
# Arms, all at the SAME EP topology so only the MoE implementation moves:
#   dtri  EP on, monokernel off, no EP patches      stock Triton EP, THE BASELINE
#   dep   EP on, ragged grid + pad-aware SiLU       the already-published EP optimization, kept only
#                                                   as an engineering control, not as the baseline
#   demk  EP on, EP-aware persistent monokernel     one launch for the whole routed MoE layer
#
# demk deliberately does NOT set VLLM_EP_MASKED_SUM or VLLM_EP_FAST_SILU: those patch the Triton
# expert-GEMM grid and the standalone activation kernel, and the monokernel has neither launch to
# patch. It carries the same three ideas inside the kernel: the counting sort is ragged so no unowned
# row is scheduled, the SiLU is fused into the down phase, and Phase 3 sums only the slots this rank
# owns, which is the pad-aware moe_sum by construction.
#
# VLLM_EP_NUM_SMS=32 is set on dep for byte-for-byte fidelity with the published c2camp `ep` arm, and
# it is INERT in this topology: use_all2all_kernels is false at DP=1/PCP=1, so there is no DeepEP
# all-to-all whose SM count it could widen. It is not claimed as a change.
#
# VLLM_USE_MOE_MONOKERNEL=0 is the clean monokernel disable. An unset VLLM_PREFILL_MONOKERNEL_SO
# would instead load a stale K_DIM=2048 default .so, which is NOT an "off" arm.
#
# FRESH LOG TAGS. Names are d1_<arm>_t<T><TAG>_b<n>, a family no existing reader globs, so no table
# row here can be joined against a boot from c2camp or ep_all6.
set -u
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# EPBASE holds the out-of-tree runtime dependencies this campaign needs on the
# import path: a ninja build of DeepEP under dep_site_v1 and, for Qwen, a
# FlashInfer wheel under fi_site. Point it at your own overlay.
EPBASE=${EPBASE:-$HOME/epbase}
export PATH=$EPBASE/ninja/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export M8PAD_DISABLE=1
export VLLM_USE_NCCL_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO

# TREE is the vLLM source tree carrying the integration in vllm_integration/.
export TREE=${TREE:-$HOME/vllm-upstream-port}
# The local snapshot directory of deepseek-ai/DeepSeek-V3. No default: an
# accidentally wrong checkpoint is a silent measurement error, not a crash.
DSNAP=${DSNAP:?set DSNAP to the DeepSeek-V3 snapshot directory}
SELFDIR=$(cd "$(dirname "$0")" && pwd)
# This script lives in research_archive/, so the harness it shares with the live
# campaigns is one level up in benchmarks/.
HARNESS=${HARNESS:-$SELFDIR/../benchmarks}
PY=${PY:-python3}
VLLM_META=${VLLM_META:-$HARNESS/vllm_dist_info_stub}
export PYTHONPATH=$VLLM_META:$EPBASE/dep_site_v1:$TREE
# The EP build is a SEPARATE artifact from the shipping TP build and a separate variable, so a TP arm
# and an EP arm can sit in the same campaign without either picking up the other's geometry. It is
# compiled at E_LOCAL=32 against a global 256 and exports ep_capable=1; the weight-load gate refuses
# it on TP-shaped weights and refuses the TP build on EP-shaped weights, in both directions.
# This build is NOT one of the three shipped kernel variants. Reproduce it by
# applying research_archive/kernel_diffs/ds3_ep.diff to kernel/deepseek_tp/src
# and building that copy, then point EPSO at the result.
EPSO=${EPSO:?set EPSO to the DeepSeek EP monokernel build (see kernel_diffs/ds3_ep.diff)}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TPD=${TPD:-8}
OUT=${OUT:-$EPBASE/d1ds3}
T=${T:-8192}
BOOTS=${BOOTS:-3}
B0=${B0:-0}
ARMS=${ARMS:-"dtri dep demk"}
REPS=${REPS:-3}
EAGER=${EAGER:-0}
CGMODE=${CGMODE:-FULL_AND_PIECEWISE}
TAG=${TAG:-}
# 0.90 at 8K and 0.95 at 16K, matching the published DeepSeek campaign, unless overridden.
if [ "$T" -ge 16384 ]; then GU=${GU:-0.95}; else GU=${GU:-0.90}; fi
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p $OUT
cd /tmp

echo "############ D1 DS3 start $(date -u +%FT%TZ) T=$T tp=$TPD gpus=$GPUS boots=$BOOTS gu=$GU eager=$EAGER cg=$CGMODE tag=$TAG"
echo "MARK preflight_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . )"
echo "MARK preflight_mem=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
echo "MARK commit=$(cd $TREE && git rev-parse --short HEAD) branch=$(cd $TREE && git rev-parse --abbrev-ref HEAD)"
echo "MARK epso=$EPSO"
echo "MARK epso_md5=$(md5sum $EPSO 2>/dev/null | cut -d" " -f1)"

# Boots are interleaved arm by arm inside each boot index so a drift in the box affects every arm of
# a pair the same way. Pairing is by boot index, which is what the reader joins on.
b=$B0
while [ $b -lt $((B0 + BOOTS)) ]; do
for arm in $ARMS; do
  # Every knob is set explicitly on every arm.
  unset VLLM_PREFILL_MONOKERNEL_SO VLLM_PREFILL_MONOKERNEL_TP_FUSE \
        VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED \
        VLLM_PREFILL_MONOKERNEL_TP_CHUNKS VLLM_PREFILL_MONOKERNEL_TP_NCOMM \
        VLLM_EP_MASKED_SUM VLLM_EP_FAST_SILU VLLM_EP_TOPK_COMPACT VLLM_EP_CG_WORST \
        VLLM_EP_NUM_SMS VLLM_EP_ROUTE_DUMP VLLM_EP_RS_CONTROL VLLM_EP_PHASE2 \
        VLLM_EP_COMB_BARRIER VLLM_EP_TIME_DUMP
  export VLLM_USE_MOE_MONOKERNEL=0
  case $arm in
    dtri) E=1 ;;
    dep)  E=1 ; export VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 VLLM_EP_NUM_SMS=32 ;;
    demk) E=1 ; export VLLM_USE_MOE_MONOKERNEL=1 VLLM_PREFILL_MONOKERNEL_SO=$EPSO ;;
    *) echo "MARK unknown arm $arm"; continue ;;
  esac
  nm=d1_${arm}_t${T}${TAG}_b${b}
  if [ -s $OUT/$nm.log ]; then echo "MARK skip $nm"; continue; fi
  # TIMEDUMP=1 turns this into the per-layer kernel probe. The prefix is derived from the boot name
  # so two arms cannot write into each other's dumps. The probe must run eager: the instrument is
  # CUDA events recorded inside the model forward, and events recorded inside a captured graph carry
  # no per-replay timing.
  if [ "${TIMEDUMP:-0}" = "1" ]; then export VLLM_EP_TIME_DUMP=$OUT/$nm; fi
  echo "MARK boot $nm arm=$arm ep=$E mono=$VLLM_USE_MOE_MONOKERNEL ms=${VLLM_EP_MASKED_SUM:-0} fs=${VLLM_EP_FAST_SILU:-0} sms=${VLLM_EP_NUM_SMS:-unset} $(date -u +%FT%TZ)"
  timeout 2400 env MODEL=$DSNAP TP=$TPD PCP=1 EP=$E TOKENS=$T REPS=$REPS GPUUTIL=$GU \
      EAGER=$EAGER CGMODE=$CGMODE CGSIZES=$T OUTTOK=1 \
      $PY $HARNESS/ep_cg_run.py > $OUT/$nm.log 2>&1
  echo "MARK done $nm rc=$? $(date -u +%FT%TZ)"
  sleep 6
done
b=$((b + 1))
done
echo "############ D1 DS3 end $(date -u +%FT%TZ)"
