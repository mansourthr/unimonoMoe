#!/bin/bash
# THE COMBINED EP+TP FOUR-ARM KERNEL CAMPAIGN. Per-layer routed MoE timing, DS3, 8 x H200.
#
# THE ARM DEFINITIONS ARE COPIED FROM c2_camp.sh CHARACTER FOR CHARACTER. The system table and this
# kernel table must not be able to disagree about what an arm is, so the four X= strings below are
# identical to the system campaign's and the topology (TP=8 / DP=1 / PCP=1 / world 8) is the same.
#
# WHY THIS PROBE IS EAGER WHILE THE SYSTEM CAMPAIGN IS CUDA-GRAPH. The instrument is CUDA events
# recorded inside the model forward. Events recorded inside a captured graph do not carry per-replay
# timings, which _ep_timer.py's own header states, so a graph-mode per-layer probe is not merely
# noisy, it is unavailable. The two campaigns therefore answer two different questions on purpose:
# the system table is the shipping mode end to end, and this table is the per-layer decomposition.
# The arm definitions are shared; the execution mode is not, and the report says so rather than
# presenting a kernel percentage as if it were the end-to-end one.
#
# THE METRIC. One span, tag "rmoe", added in moe_runner.py, bracketing MoERunner.forward from just
# before _forward_entry to just after _maybe_reduce_final_output. That is the same logical work in
# every arm: router logits in hand through to the complete reduced MoE output, INCLUDING the output
# collective wherever it lives. The pre-existing "layer" span cannot be used for this, because it
# sits in the modular-kernel branch and never fires when the monokernel accepts a call.
#
# Prefill only, one generated token, so each rep is exactly one forward pass and the ordered dump
# chunks cleanly at 58 routed layers per pass. Rep 0 is discarded as warmup by the reader.
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
OUT=${OUT:-$EPBASE/c3kern}
SHIPSO=${SHIPSO:-$KERNEL/deepseek_tp/build/libprefill_mono.so}
T=${T:-8192}
GU=${GU:-0.90}
BOOTS=${BOOTS:-4}
REPS=${REPS:-3}
ARMS=${ARMS:-"tri ep tp eptp"}
mkdir -p $OUT
cd /tmp

echo "############ C3 KERN start $(date -u +%FT%TZ) T=$T GU=$GU BOOTS=$BOOTS REPS=$REPS"
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
    nm=c3_${arm}_t${T}_b${b}
    if [ -s $OUT/$nm.log ]; then echo "MARK skip $nm"; continue; fi
    echo "MARK boot $nm $(date -u +%FT%TZ)"
    env $X MODEL=$DSNAP VLLM_EP_TIME_DUMP=$OUT/$nm \
        TP=8 PCP=1 EP=$E TOKENS=$T REPS=$REPS GPUUTIL=$GU \
        EAGER=1 CGMODE=NONE OUTTOK=1 \
        $PY $HARNESS/ep_cg_run.py > $OUT/$nm.log 2>&1
    echo "MARK done $nm rc=$? $(date -u +%FT%TZ)"
    echo "MARK dumps $nm n=$(ls $OUT/$nm.rank*.json 2>/dev/null | grep -c .)"
    sleep 6
  done
done
echo "############ C3 KERN end $(date -u +%FT%TZ)"
