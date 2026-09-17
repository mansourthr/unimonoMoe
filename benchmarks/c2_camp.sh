#!/bin/bash
# THE COMBINED EP+TP FOUR-ARM SYSTEM CAMPAIGN. Prefill / TTFT only, DS3, 8 x H200.
#
# ONE MATCHED CAMPAIGN. Every arm runs at TP=8 / DP=1 / PCP=1 / world 8, same model, same
# sonnet.txt prompt tiled to exactly T tokens, one generated token, prefix caching disabled,
# seed 0, the same gpu_memory_utilization, and the SAME execution mode (CUDA graph
# FULL_AND_PIECEWISE with the benchmark shape in the capture list). Only the expert-parallel
# flag and the optimization flags move between arms. That is the strongest match this runtime
# allows, because fused_moe/config.py makes MoE-level EP and MoE-level TP mutually exclusive:
# under enable_expert_parallel it sets ep_size = dp*pcp*tp and tp_size = 1.
#
#   tri   EP off, monokernel off      Triton fused_experts + external NCCL all-reduce
#   ep    EP off->on, monokernel off  Triton experts + ragged grid + pad-aware SiLU
#   tp    EP off, monokernel + fuse   persistent monokernel, collective released in Phase 3
#   eptp  EP on,  monokernel + fuse   both optimization sets requested in one boot
#
# INTERLEAVED AND PAIRED BY BOOT INDEX. All four arms run inside boot index b before b+1 starts,
# so a slow patch of the machine hits every arm of that index rather than one arm's whole run.
# The reader pairs on the common boot index and never pools across campaigns.
#
# THE FIXED CONFIGURATION IS IN THE SOURCE, NOT CHOSEN HERE. _FUSED_CHUNKS_BY_TOKENS in
# monokernel_tp.py is {8192: 4, 16384: 8} with _FUSED_NCOMM = 14, and there is no environment
# override, so no per-boot or per-process best-of is possible. Read back per boot as
# `ncomm=14 sizes=[...] fold_shared=True`.
#
# VLLM_USE_MOE_MONOKERNEL=0 is the clean monokernel disable. An unset VLLM_PREFILL_MONOKERNEL_SO
# would instead load a stale K_DIM=2048 default .so, which declines on the wrong geometry and is
# NOT an "off" arm.
#
# VLLM_EP_NUM_SMS=32 is set on the EP arms for fidelity with the shipping EP configuration, but it
# is INERT in this topology: use_all2all_kernels is false at DP=1/PCP=1, so there is no DeepEP
# all-to-all whose SM count it could widen. It is not claimed as a change in the report.
set -u
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# EPBASE holds the out-of-tree runtime dependencies this campaign needs on the
# import path: a ninja build of DeepEP under dep_site_v1 and, for Qwen, a
# FlashInfer wheel under fi_site. Point it at your own overlay.
EPBASE=${EPBASE:-$HOME/epbase}
export PATH=$EPBASE/ninja/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_FLASHINFER_SAMPLER=0
export M8PAD_DISABLE=1
export VLLM_USE_NCCL_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset VLLM_EP_ROUTE_DUMP VLLM_EP_RS_CONTROL VLLM_EP_TOPK_DUMP VLLM_EP_TOPK_COMPACT \
      VLLM_EP_PHASE2 VLLM_EP_TIME_DUMP VLLM_EP_CG_WORST VLLM_EP_MASKED_SUM \
      VLLM_EP_FAST_SILU VLLM_EP_NUM_SMS VLLM_PREFILL_MONOKERNEL_SO \
      VLLM_PREFILL_MONOKERNEL_TP_FUSE VLLM_USE_MOE_MONOKERNEL \
      VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED
# TREE is the vLLM source tree carrying the integration in vllm_integration/.
export TREE=${TREE:-$HOME/vllm-upstream-port}
# The local snapshot directory of deepseek-ai/DeepSeek-V3. No default: an
# accidentally wrong checkpoint is a silent measurement error, not a crash.
DSNAP=${DSNAP:?set DSNAP to the DeepSeek-V3 snapshot directory}
SELFDIR=$(cd "$(dirname "$0")" && pwd)
HARNESS=${HARNESS:-$SELFDIR}
PY=${PY:-python3}
VLLM_META=${VLLM_META:-$HARNESS/vllm_dist_info_stub}
# The compiled kernels, as written by kernel/build.sh <variant>. Overridable so a
# binary built elsewhere can be substituted without editing the script.
KERNEL=${KERNEL:-$SELFDIR/../kernel}
export PYTHONPATH=$VLLM_META:$EPBASE/dep_site_v1:$TREE
OUT=${OUT:-$EPBASE/c2camp}
SHIPSO=${SHIPSO:-$KERNEL/deepseek_tp/build/libprefill_mono.so}
T=${T:-8192}
GU=${GU:-0.90}
BOOTS=${BOOTS:-6}
REPS=${REPS:-3}
ARMS=${ARMS:-"tri ep tp eptp"}
mkdir -p $OUT
cd /tmp

echo "############ C2 CAMP start $(date -u +%FT%TZ) T=$T GU=$GU BOOTS=$BOOTS REPS=$REPS"
echo "MARK preflight_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . )"

for b in $(seq 0 $((BOOTS - 1))); do
  for arm in $ARMS; do
    case $arm in
      tri)  E=0 ; X="VLLM_USE_MOE_MONOKERNEL=0" ;;
      ep)   E=1 ; X="VLLM_USE_MOE_MONOKERNEL=0 VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 VLLM_EP_NUM_SMS=32" ;;
      tp)   E=0 ; X="VLLM_PREFILL_MONOKERNEL_SO=$SHIPSO VLLM_PREFILL_MONOKERNEL_TP_FUSE=1" ;;
      eptp) E=1 ; X="VLLM_PREFILL_MONOKERNEL_SO=$SHIPSO VLLM_PREFILL_MONOKERNEL_TP_FUSE=1 VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 VLLM_EP_NUM_SMS=32" ;;
      *) echo "MARK unknown arm $arm"; continue ;;
    esac
    nm=c2_${arm}_t${T}_b${b}
    if [ -s $OUT/$nm.log ]; then echo "MARK skip $nm"; continue; fi
    echo "MARK boot $nm $(date -u +%FT%TZ)"
    env $X MODEL=$DSNAP TP=8 PCP=1 EP=$E TOKENS=$T REPS=$REPS GPUUTIL=$GU \
        EAGER=0 CGMODE=FULL_AND_PIECEWISE CGSIZES=$T OUTTOK=1 \
        $PY $HARNESS/ep_cg_run.py > $OUT/$nm.log 2>&1
    echo "MARK done $nm rc=$? $(date -u +%FT%TZ)"
    sleep 6
  done
done
echo "############ C2 CAMP end $(date -u +%FT%TZ)"
