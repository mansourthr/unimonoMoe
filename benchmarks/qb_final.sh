#!/bin/bash
# Qwen3.5-35B-A3B-FP8 FINAL campaign. Everything measured here is a report number.
#
# FROZEN CONFIGURATION, one per workload, decided before this script ran and not
# touched again. No per-boot or per-process best-of selection is applied downstream.
#
#   TP, after optimization (arm `fuse`)
#     VLLM_USE_MOE_MONOKERNEL=1
#     VLLM_PREFILL_MONOKERNEL_SO=kernel/qwen_tp/build/libprefill_mono.so
#     VLLM_PREFILL_MONOKERNEL_TP_FUSE=1
#     chunks {8192: 4, 16384: 8}   ncomm 14   fold_shared 1
#   These three release-geometry values are the DeepSeek-V3 defaults, kept because the
#   paired confirmation sweep measured every candidate change inside noise on Qwen:
#   chunks 2 vs 4 at 8192 was -0.34% with 1 of 3 boots ahead, chunks 4 vs 8 at 16384 was
#   +0.02% with 1 of 3 boots ahead, ncomm 8 / 14 / 20 / 26 spanned 1.8% with no ordering,
#   and fold off vs on was 0.4%. The knobs are therefore NOT overridden here at all, so
#   the frozen arm runs the code's own defaults and the log line proves which values ran.
#
#   EP, after optimization (arm `em1f`)
#     VLLM_EP_MASKED_SUM=1     ragged expert GEMM grid, pad-aware moe_sum
#     VLLM_EP_FAST_SILU=2      pad-aware fused activation and per-block quantization
#   VLLM_EP_NUM_SMS, VLLM_EP_TOPK_COMPACT and VLLM_EP_CG_WORST are not set because they
#   are inert on Qwen: all three are read only inside the DeepEP prepare/finalize and
#   all2all manager, and Qwen resolves to the allgather-reducescatter manager.
#
# WHY TWO MODES. The system rows run under CUDA graph, because Qwen captures the full
# prefill in every arm at both workloads and graph is the faster and therefore the
# honest configuration. The kernel rows run eager, because the per-layer instrument is
# CUDA events recorded inside the model forward and events recorded inside a captured
# graph carry no per-replay timing. The two tables are never compared to each other.
set -u
Q=${Q:-$(cd "$(dirname "$0")" && pwd)/q4_qwen.sh}
L=${L:-/tmp/qb_final.out}
A="tri mono fuse etri em1f"
: > $L

step() { echo "######## FINAL $* $(date -u +%FT%TZ)"; }

# System, CUDA graph. 5 arms interleaved inside each of boot indices 0 1 2.
step system T=8192
env TAG=_sys T=8192  BOOTS=3 EAGER=0 CGMODE=FULL_AND_PIECEWISE ARMS="$A" bash $Q >> $L 2>&1
step system T=16384
env TAG=_sys T=16384 BOOTS=3 EAGER=0 CGMODE=FULL_AND_PIECEWISE ARMS="$A" bash $Q >> $L 2>&1

# Kernel, eager, per-layer dumps. FOUR boot indices, not three: each rank writes its
# dump from an atexit hook at worker exit, and the smoke boot lost one rank's file out of
# twenty when a worker was reaped during teardown. A boot missing any rank's dump is
# excluded whole, because a three-rank mean is a different statistic from a four-rank
# mean, so the extra boot index is margin against an exclusion rather than a fourth
# sample to choose from.
step kernel T=8192
env TIMEDUMP=1 TAG=_krn T=8192  BOOTS=4 EAGER=1 CGMODE=NONE ARMS="$A" bash $Q >> $L 2>&1
step kernel T=16384
env TIMEDUMP=1 TAG=_krn T=16384 BOOTS=4 EAGER=1 CGMODE=NONE ARMS="$A" bash $Q >> $L 2>&1

echo "######## FINAL end $(date -u +%FT%TZ)"
