#!/bin/bash
# Qwen3.5-35B-A3B-FP8 prefill / TTFT campaign at the legal 4-rank topology.
#
# Why 4 ranks and not 8. The gated shared expert's down_proj is sharded on its input dimension,
# shared_expert_intermediate_size / world = 512 / world, and the fp8 checkpoint carries
# weight_block_size [128, 128], so validate_fp8_block_shape rejects any world above 512/128 = 4.
# At --tp 8 every rank dies at weight load with "Weight input_size_per_partition = 64 is not
# divisible by weight quantization block_k = 128", with expert parallel on or off. 4 is therefore
# the widest legal Qwen3.5 world on this box, not a choice.
#
# Arms. Each is one row of the report's three-row tables, per parallel mode.
#   tri    EP off, monokernel off                       Triton baseline, tensor parallel
#   mono   EP off, monokernel on                        Before optimization, TP: persistent kernel,
#                                                       external all-reduce still exposed
#   fuse   EP off, monokernel on, fused TP collective   After optimization, TP
#   etri   EP on,  monokernel off, no EP patches        Triton baseline / Before optimization, EP
#   em1f   EP on,  ragged grid + pad-aware SiLU         After optimization, EP
#   emk    EP on,  EP-aware persistent monokernel       The EP monokernel, one launch for the whole
#                                                       routed layer. VLLM_EP_MASKED_SUM and
#                                                       VLLM_EP_FAST_SILU are deliberately NOT set:
#                                                       they patch the Triton grid and the standalone
#                                                       activation kernel, and the monokernel has
#                                                       neither. What it carries INSIDE the kernel is
#                                                       the same three ideas: the counting sort is
#                                                       ragged so no unowned row is ever scheduled
#                                                       (no padded tile exists to skip), the SiLU is
#                                                       fused into the down phase so there is no
#                                                       separate activation launch, and Phase 3 sums
#                                                       only the slots this rank owns, which is the
#                                                       pad-aware moe_sum by construction.
#
# GPUS defaults to 4,5,6,7. GPU0 carries an unrelated tenant's inference server, and an idle GPU is
# a correctness precondition here, not a performance preference.
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
# Qwen3.5's Gated DeltaNet linear-attention layers resolve their prefill backend through
# additional_config and default to a flashinfer path that JIT compiles and needs the cutlass python
# DSL, which is not installed. Pinned to triton and held identical in every arm, so it cannot move a
# comparison.
export ADDCFG=${ADDCFG:-gdn_prefill_backend=triton}

# TREE is the vLLM source tree carrying the integration in vllm_integration/.
export TREE=${TREE:-$HOME/vllm-upstream-port}
SELFDIR=$(cd "$(dirname "$0")" && pwd)
HARNESS=${HARNESS:-$SELFDIR}
PY=${PY:-python3}
VLLM_META=${VLLM_META:-$HARNESS/vllm_dist_info_stub}
# The compiled kernels, as written by kernel/build.sh <variant>. Overridable so a
# binary built elsewhere can be substituted without editing the script.
KERNEL=${KERNEL:-$SELFDIR/../kernel}
export PYTHONPATH=$VLLM_META:$EPBASE/dep_site_v1:$EPBASE/fi_site:$TREE
# The local snapshot directory of Qwen/Qwen3.5-35B-A3B-FP8. No default: an
# accidentally wrong checkpoint is a silent measurement error, not a crash.
QSNAP=${QSNAP:?set QSNAP to the Qwen3.5-35B-A3B-FP8 snapshot directory}
QSO=${QSO:-$KERNEL/qwen_tp/build/libprefill_mono.so}
# The EP build is a SEPARATE artifact and a separate variable, so a TP arm and an EP arm can sit in
# the same campaign without either one picking up the other's geometry. It is compiled at E_LOCAL=64
# against a global 256 and exports ep_capable=1; the weight-load gate refuses to enable it on
# TP-shaped weights and refuses the TP build on EP-shaped weights, in both directions.
EPSO=${EPSO:-$KERNEL/qwen_ep/build/libprefill_mono.so}

GPUS=${GPUS:-4,5,6,7}
TPD=${TPD:-4}
OUT=${OUT:-$EPBASE/q4qwen}
T=${T:-8192}
BOOTS=${BOOTS:-1}
B0=${B0:-0}
ARMS=${ARMS:-"tri mono fuse etri em1f"}
REPS=${REPS:-3}
EAGER=${EAGER:-1}
CGMODE=${CGMODE:-NONE}
TAG=${TAG:-}
# 0.90 at 8K and 0.95 at 16K, matching the DeepSeek campaign, unless overridden.
if [ "$T" -ge 16384 ]; then GU=${GU:-0.95}; else GU=${GU:-0.90}; fi
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p $OUT
cd /tmp

echo "############ Q4 QWEN start $(date -u +%FT%TZ) T=$T tp=$TPD gpus=$GPUS boots=$BOOTS gu=$GU"
echo "MARK preflight_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . )"
echo "MARK commit=$(cd $TREE && git rev-parse --short HEAD) branch=$(cd $TREE && git rev-parse --abbrev-ref HEAD)"
echo "MARK so=$QSO"
echo "MARK epso=$EPSO"
echo "MARK epso_md5=$(md5sum $EPSO 2>/dev/null | cut -d" " -f1)"

# Boots are interleaved arm by arm inside each boot index so that a drift in the box affects every
# arm of a pair the same way. Pairing is by boot index, which is what the reader joins on.
b=$B0
while [ $b -lt $((B0 + BOOTS)) ]; do
for arm in $ARMS; do
  # Every knob is set explicitly on every arm. Leaving VLLM_USE_MOE_MONOKERNEL unset is not the off
  # switch: envs.py carries a hardcoded default path for VLLM_PREFILL_MONOKERNEL_SO, and on Qwen at
  # TP=4 that default build (N_HALF=512) exports neither n_up nor n_half, so the geometry gate
  # skipped both width checks, enabled the kernel at the wrong stride, and the boot died with an
  # illegal memory access.
  unset VLLM_PREFILL_MONOKERNEL_SO VLLM_PREFILL_MONOKERNEL_TP_FUSE \
        VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED \
        VLLM_PREFILL_MONOKERNEL_TP_CHUNKS VLLM_PREFILL_MONOKERNEL_TP_NCOMM \
        VLLM_EP_MASKED_SUM VLLM_EP_FAST_SILU VLLM_EP_TOPK_COMPACT VLLM_EP_CG_WORST \
        VLLM_EP_NUM_SMS VLLM_EP_ROUTE_DUMP VLLM_EP_RS_CONTROL VLLM_EP_PHASE2 \
        VLLM_EP_COMB_BARRIER VLLM_EP_TIME_DUMP
  export VLLM_USE_MOE_MONOKERNEL=0
  case $arm in
    tri)  E=0 ;;
    mono) E=0 ; export VLLM_USE_MOE_MONOKERNEL=1 VLLM_PREFILL_MONOKERNEL_SO=$QSO ;;
    fuse) E=0 ; export VLLM_USE_MOE_MONOKERNEL=1 VLLM_PREFILL_MONOKERNEL_SO=$QSO \
                       VLLM_PREFILL_MONOKERNEL_TP_FUSE=1 ;;
    etri) E=1 ;;
    em1f) E=1 ; export VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 ;;
    emk)  E=1 ; export VLLM_USE_MOE_MONOKERNEL=1 VLLM_PREFILL_MONOKERNEL_SO=$EPSO ;;
    # EP tuning arms. The two knobs are independent code paths, so the split is what
    # says which one carries the win: MASKED_SUM changes the expert GEMM grid to skip
    # padded rows and makes moe_sum pad aware, FAST_SILU changes the activation and
    # its per-block quantization. Mode 1 is the fast activation WITHOUT the pad-aware
    # variant; mode 2 additionally passes topk_ids and expert_map so the activation
    # skips rows no local expert owns.
    ems)  E=1 ; export VLLM_EP_MASKED_SUM=1 ;;
    efs1) E=1 ; export VLLM_EP_FAST_SILU=1 ;;
    efs2) E=1 ; export VLLM_EP_FAST_SILU=2 ;;
    em1s1) E=1 ; export VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=1 ;;
    *) echo "MARK unknown arm $arm"; continue ;;
  esac
  # Optional per-arm overrides for the tuning phase only. FOLD and NCOMM/CHUNKS are exported by the
  # caller and are recorded in the log line below so a tuning boot can never be mistaken for a
  # frozen-config boot.
  if [ -n "${FOLD:-}" ]; then export VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED=$FOLD; fi
  # CHUNKS and NCOMM override the fused-TP release geometry, which was measured on
  # DeepSeek-V3 at hidden 7168 / world 8. Qwen runs hidden 2048 / world 4, so a chunk
  # carries 3.5x fewer bytes and there are half as many peers: the DeepSeek values are
  # a starting point, not a result. Unset means the DeepSeek table, unchanged.
  if [ -n "${CHUNKS:-}" ]; then export VLLM_PREFILL_MONOKERNEL_TP_CHUNKS=$CHUNKS; fi
  if [ -n "${NCOMM:-}" ]; then export VLLM_PREFILL_MONOKERNEL_TP_NCOMM=$NCOMM; fi
  nm=q4_${arm}_t${T}${TAG}_b${b}
  if [ -s $OUT/$nm.log ]; then echo "MARK skip $nm"; continue; fi
  # TIMEDUMP=1 turns this into the per-layer kernel probe. The prefix is derived from the boot name
  # so two arms cannot write into each other's dumps. The probe must run eager: the instrument is
  # CUDA events recorded inside the model forward, and events recorded inside a captured graph carry
  # no per-replay timing.
  if [ "${TIMEDUMP:-0}" = "1" ]; then export VLLM_EP_TIME_DUMP=$OUT/$nm; fi
  echo "MARK boot $nm arm=$arm ep=$E mono=$VLLM_USE_MOE_MONOKERNEL fuse=${VLLM_PREFILL_MONOKERNEL_TP_FUSE:-0} fold=${VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED:-default} chunks=${VLLM_PREFILL_MONOKERNEL_TP_CHUNKS:-default} ncomm=${VLLM_PREFILL_MONOKERNEL_TP_NCOMM:-default} ms=${VLLM_EP_MASKED_SUM:-0} fs=${VLLM_EP_FAST_SILU:-0} $(date -u +%FT%TZ)"
  timeout 2400 env MODEL=$QSNAP \
      TP=$TPD PCP=1 EP=$E TOKENS=$T REPS=$REPS GPUUTIL=$GU \
      EAGER=$EAGER CGMODE=$CGMODE OUTTOK=1 \
      $PY $HARNESS/ep_cg_run.py > $OUT/$nm.log 2>&1
  echo "MARK done $nm rc=$? $(date -u +%FT%TZ)"
  sleep 6
done
b=$((b + 1))
done
echo "############ Q4 QWEN end $(date -u +%FT%TZ)"
