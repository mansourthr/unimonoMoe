// DeepSeek-V3 prefill MoE monokernel, tensor parallel (TP=8 shard).
//
// Integrated prefill MoE kernel (H200 / sm_90a): the verified Phase 0-1
// routing + sort pipeline from prefill_kernel.cu fused with the verified
// Phase 2 WGMMA up->SiLU->quantize->down->atomicAdd compute chain from
// wgmma_spike_ref.cu, plus a Phase 3 fp32->bf16 cast.
//
// Single persistent launch, phases separated by software grid barriers
// (reused verbatim from the decode monokernel's moe_grid_barrier.h):
//
//   Phase 0 : TopK routing (softmax topk, one warp per token)
//   -- grid barrier #1 --
//   Phase 1a: per-expert histogram (atomicAdd in GM)
//   -- grid barrier #2 --
//   Phase 1b: block 0 prefix-sums the histogram -> expert_offsets[E+1],
//             seeds per-expert write cursors, builds the tile schedule
//   -- grid barrier #3 --
//   Phase 1c: scatter flattened topk indices into sorted_token_ids
//             via atomicAdd on the per-expert write cursors
//   -- grid barrier #4 --
//   Phase 2 : persistent-block tile dispatch running the full WGMMA
//             compute chain per tile:
//               up-proj (WGMMA, TMA weight) -> SiLU (fp32 SHM)
//               -> quantize fp32->FP8 with 128-col block scales
//               -> down-proj (WGMMA, manual-swizzle weight load, B from
//                  on-chip SHM intermediate)
//               -> x topk_weight -> atomicAdd into output_accum[BS, H_DIM]
//   -- grid barrier #5 --
//   Phase 3 : fp32 -> bf16 cast: output_accum -> bf16_output
//
// Phase 0-1 uses NO dynamic SHM (everything goes through the GMEM
// Workspace struct).  Phase 2 uses dynamic SHM sized by the host for its
// peak (weight tile + mbarrier + act slices + fp32 SiLU temp + fp8
// intermediate + scales + down_act staging).
//
// sorted_token_ids uses the same semantics as vllm's moe_align_block_size:
// each entry is an index into the FLATTENED topk array, i.e.
// token_index * TOP_K + k.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// fp16-oa scaling: keep the per-expert partial inside fp16's NORMAL range so the
// fp16 round keeps its full 10-bit mantissa. Powers of two, so partial*S and
// sum*(1/S) are EXACT in fp32 -- the rescaling contributes no error of its own.
//
// S was 2^16, chosen from partials sampled at layers 0 and 39 of the real
// checkpoint (max 1.6e-3, which looked like 40x headroom). Running the kernel as
// the MoE layer of the real model showed that sample was not representative: the
// deep layers carry far larger partials. Measured over all 40 layers and 7 prompts
// with the fp32-oa build (whose output_accum really is fp32, so its numbers are
// ground truth):
//     layer  0 max 3.55e-2      layer 33 max 4.35e-1
//     layer 30 max 2.00e-1      layer 38 max 1.08e+0
//     layer 24 max 3.82e-2      layer 39 max 1.30e+0   <- global max
// The ceiling is 65504, so S must be < 65504/1.2959 = 50548. At S=2^16 the largest
// partial scales to 84927 and rounds to +Inf, which poisons the residual stream and
// makes the NEXT layer's router logits NaN -- observed as an illegal memory access
// at layer 39, because softmax over NaN yields an undefined top-8 index and an
// out-of-range expert index is an out-of-range weight offset.
//
// S = 2^12 leaves 12.3x headroom on the measured global max. A smaller S costs
// nothing in accuracy: fp16 carries all 10 mantissa bits anywhere in its normal
// range, so only staying NORMAL matters, not being centred. The p01 partial
// (1.53e-5) scales to 2.8e-2, still 463x above fp16's min normal 6.10e-05, so the
// small end does not go subnormal either.
// SUPERSEDED by bf16-oa, below. The reasoning above is correct for Qwen3.5 and
// is kept because it records how the constant was sized and what it cost to get
// wrong. It does not survive a change of model: MEASURED on MiniMax-M2, the peak
// partial is 1339.5477 and 44 of 62 layers exceed 65504/2^12 = 15.9921875 at the
// real prompt length, against the 1.2959 global max these numbers come from.
// Modelling that ceiling reproduces this kernel's output at M2 layer 5 to 1.4e-2
// and 5.3e-3 relative, where the unclamped model is off by 8.96 and 1.94, and it
// predicts exactly which 2 of 1594 tokens fail with no false positives and no
// false negatives.
//
// bf16-oa: store the partial UNSCALED as bf16. bf16 has the exponent range of
// fp32, so no model can reach a ceiling and there is no scale left to size
// wrongly. The choice was measured against a fp64 reference on M2 layer 5's real
// partials (worst-token relative L2):
//     fp16 S=2^12   9.3809e-01   2 tokens short   <- what shipped
//     fp16 S=2^5    2.3358e-04   0 tokens short   <- largest S clearing THIS prompt
//     bf16 no scale 1.7527e-03   0 tokens short
//     fp32 no scale 7.8613e-08   0 tokens short
// S=2^5 works here and is rejected anyway: it leaves 1.53x headroom on a single
// prompt of a single model, which is the same fit that produced this bug. bf16
// costs 8 mantissa bits instead of 10, i.e. 1.75e-3 worst-token error, which is
// 8x below the 1.45e-2 at which the Qwen3.5 positive control already agrees with
// vLLM's own baseline. fp32 would be exact but doubles phase-3 read traffic;
// bf16 keeps the 2-byte store and the same 32 B vector loads.
//
// The clamp is kept and retargeted to bf16's max normal. It does not fire on any
// measured data, and it preserves what the fp16 clamp was for: an unforeseen
// partial degrades to a bounded value rather than an Inf that poisons the next
// layer's router logits.
#define OA_BF16_CLAMP     3.38953139e38f      // bf16 max normal

#include <cstdint>
#include <cstdio>
#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "../../moe_monokernel/src/ptx_utils.h"
#include "../../moe_monokernel/src/moe_grid_barrier.h"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
using namespace moe_monokernel;

namespace {

// ======================== Launch shape ========================
constexpr uint32_t GRID_SIZE = 132;   // persistent blocks, <= 132 SMs (H200)
constexpr uint32_t BLOCK_SIZE = 256;  // threads per block
constexpr int GRID = static_cast<int>(GRID_SIZE);
constexpr int BLOCK = static_cast<int>(BLOCK_SIZE);

// ======================== Problem constants ========================
constexpr int E = 256;                     // experts
constexpr int NUM_EXPERTS = E;
constexpr int TOP_K = 8;
constexpr int K_DIM = 7168;                // hidden dim (up-proj K), DeepSeek-V3
constexpr int N_HALF = 256;                 // post-SiLU width, TP=8 shard of 2048
constexpr int N_UP = 2 * N_HALF;            // gate+up fused, DERIVED so it cannot desynchronize
constexpr int H_DIM = 7168;                // down-proj output width
constexpr int N_KBLK = K_DIM / 128;        // 128-wide K blocks, up GEMM
constexpr int N_WROW = N_UP / 128;         // 128-wide N blocks, up weights
constexpr int D_KBLK = N_HALF / 128;       // 128-wide K blocks, down GEMM
constexpr int D_COL_TILES = H_DIM / 128;   // 128-wide N tiles, down output
constexpr int D_WROW = H_DIM / 128;        // 128-wide N blocks, down weights

constexpr int WARPS_PER_BLOCK = BLOCK / 32;
constexpr int LOGITS_PER_LANE = NUM_EXPERTS / 32;  // 8

// Router selection. 0 = softmax over the logits, which is what Qwen3.5 uses and
// what every measurement before this one was taken on. 1 = sigmoid scores with
// an optional per-expert correction bias added for selection only. 2 = mode 1
// restricted to the TOPK_GROUP highest scoring expert groups, each group scored
// by the sum of its top 2 biased scores.
static constexpr int ROUTER_MODE = 2;
static constexpr int N_GROUP = 8;
static constexpr int TOPK_GROUP = 4;
// Register slot j spans experts [32j, 32j+32), which is exactly the group the
// reference forms when it reshapes a score row into N_GROUP by 32. That is what
// makes a group score one warp reduction instead of a shared memory pass, and it
// only holds when the group count equals the per-lane logit count.
static_assert(ROUTER_MODE != 2 || N_GROUP == LOGITS_PER_LANE,
              "grouped routing needs N_GROUP == LOGITS_PER_LANE (32 experts per group)");
static_assert(ROUTER_MODE != 2 || (TOPK_GROUP > 0 && TOPK_GROUP <= N_GROUP),
              "TOPK_GROUP must select at least one group and at most N_GROUP");
static_assert(ROUTER_MODE >= 0 && ROUTER_MODE <= 2, "unknown ROUTER_MODE");

// ======================== WGMMA / TMA constants ========================
constexpr uint32_t K_CHUNK = 16;
constexpr uint32_t T_PAD = 9;
constexpr uint32_t WGMMAS_PER_KS = 4;
constexpr uint64_t W_LBO = 16ULL;
constexpr uint64_t W_SBO = 1024ULL;
constexpr uint32_t W_SWZ = 1u;
constexpr uint64_t ACT_LBO = T_PAD * K_CHUNK;
constexpr uint64_t ACT_SBO = ACT_LBO;

constexpr int W_SHM_BYTES = 128 * 128;              // 16KB weight tile
constexpr int ACT_KS_BYTES = 8 * T_PAD * K_CHUNK;   // 1152B per act slice

// Was 64, the m64 WGMMA's own M. A taller tile is now issued as M_SUBS
// sub-tiles of 64 rows, so the ceiling is set by the accumulator register
// budget rather than by the instruction shape.
constexpr int MAX_BLOCK_M = 128;

// Worst case tiles: sum_e ceil(count_e / BLOCK_M) <= numel/BLOCK_M + E.
// BS=2048, K=8, BLOCK_M=32 => 512 + 256 = 768.  4096 leaves ample slack.
constexpr int MAX_TILES = 32768;

// Number of grid-barrier call sites in the kernel.  Each site gets its
// own dedicated 2-slot counter pair, zero-initialized by the host, and
// is used exactly once per launch.  Rationale: the ping-pong barrier
// leaves the exited slot at 0x80000000; if a slot is REUSED (call N+2 on
// the same pair) a non-seed block racing ahead of the seeder observes
// the stale bit 31 and exits the barrier early.  In this kernel the
// seeder (block 0) is the long pole at barrier #3 (it serially runs
// Phase 1b), which makes that race fire deterministically.  One
// fresh-zero pair per site removes slot reuse entirely, so the stale
// marker can never exist.
constexpr int NUM_BARRIER_SITES = 4;

// GM scratchpad.  All members are 4-byte; no padding anywhere.
// Chunk cap for the fused epilogue. Sizes the per-chunk arrival counters, and the
// host reads it back through prefill_wgmma_tp_max_c().
static constexpr int TP_MAX_C = 16;

struct Workspace {
  uint32_t barrier[NUM_BARRIER_SITES][2];  // one fresh pair per site
  int32_t expert_counts[NUM_EXPERTS];      // Phase 1a histogram
  int32_t expert_offsets[NUM_EXPERTS + 1]; // Phase 1b exclusive prefix sum
  int32_t write_ptrs[NUM_EXPERTS];         // Phase 1c scatter cursors
  int32_t total_tiles;                     // Phase 1b tile schedule size
  int32_t tile_expert[MAX_TILES];          // per-tile owning expert
  int32_t tile_row_start[MAX_TILES];       // per-tile row start (global sorted idx)
  // Tensor-parallel epilogue coordination. Zeroed by the launcher's existing
  // cudaMemsetAsync over sizeof(Workspace), so the fused path adds no memset of
  // its own to the timed stream and needs no cross-launch epoch tag.
  uint32_t tp_ready[TP_MAX_C];             // arrived compute blocks, per chunk
  uint32_t tp_done_local;                  // comm blocks past their last shard
  uint32_t tp_done_flag;                   // set once the final rank barrier is past
  uint32_t tp_seq_pub;                     // this launch's sequence, published once
};

// ===== Tensor-parallel epilogue: chunked one-shot all-reduce =====
//
// Why block specialisation and not a second kernel: this kernel is persistent at
// one block per SM across every SM, so a separately launched communication kernel
// cannot obtain an SM until these blocks retire, which is exactly when the
// overlap window has closed. Blocks already resident have no such problem. There
// is no grid barrier after sync_grid() #5, so communication blocks that spin
// cannot hold up a compute block. Nothing here adds or removes a kernel.
//
// The reduce is the standard one-shot: a chunk is partitioned over all
// world*ncomm communication blocks, and each shard is reduced across every rank
// through the multicast alias and stored back to every rank, so each byte is
// summed exactly once.
constexpr size_t TP_MM = 8;      // bf16 per multimem op, 16 B
constexpr size_t TP_ALIGN = 32;  // shard granularity, 32 x 16 B = one warp, 512 B

__device__ __forceinline__ uint32_t tp_ld_relaxed(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Arrive on every peer's slot, then wait for every peer on my own slots. `seq`
// increases every launch, so the pads never need clearing between launches, which
// would otherwise race a peer that is already inside the kernel.
__device__ __forceinline__ void tp_rank_barrier(uint32_t** pads, int rank,
                                                int world, int slot,
                                                uint32_t seq) {
  for (int r = 0; r < world; ++r) {
    uint32_t* p = pads[r] + (size_t)slot * world + rank;
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :: "l"(p), "r"(seq) : "memory");
  }
  for (int r = 0; r < world; ++r) {
    const uint32_t* p = pads[rank] + (size_t)slot * world + r;
    uint32_t v;
    do {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                   : "=r"(v) : "l"(p) : "memory");
    } while (v < seq);
  }
}

// acc::f32 accumulates the ranks in fp32 before rounding back to bf16. Without it
// the sum is taken in bf16 and the result differs from the reference collective by
// up to 0.0625 absolute, MEASURED.
__device__ __forceinline__ void tp_mm_ld(uint32_t* a, uint32_t& x0, uint32_t& x1,
                                         uint32_t& x2, uint32_t& x3) {
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
               "{%0,%1,%2,%3}, [%4];"
               : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "l"(a) : "memory");
}

// .v4.f32 is the only vector spelling ptxas accepts for multimem.st. The payload
// is the raw 16 B that came out of ld_reduce, so the type is a bit-width tag only
// and nothing is reinterpreted.
__device__ __forceinline__ void tp_mm_st(uint32_t* a, uint32_t x0, uint32_t x1,
                                         uint32_t x2, uint32_t x3) {
  asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(a), "r"(x0), "r"(x1), "r"(x2), "r"(x3) : "memory");
}

// One-shot all-reduce over [v_lo, v_hi) in units of 8 bf16, strided by the block.
// UNROLL independent units per thread, every load issued before any store, so the
// block keeps UNROLL loads in flight instead of one; the reduce reads across
// NVLink, where one dependent load per thread leaves the link waiting on latency.
// The stride stays BLOCK_SIZE so every pass is fully coalesced, and the last pass
// is predicated rather than peeled so the tail keeps the same loads in flight.
__device__ __forceinline__ void tp_mm_reduce(__nv_bfloat16* mc, size_t v_lo,
                                             size_t v_hi) {
  constexpr int UNROLL = 4;
  constexpr size_t STEP = (size_t)UNROLL * BLOCK_SIZE;
  for (size_t base = v_lo + threadIdx.x; base < v_hi; base += STEP) {
    uint32_t* a[UNROLL];
    uint32_t x[UNROLL][4];
    bool live[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const size_t v = base + (size_t)u * BLOCK_SIZE;
      live[u] = v < v_hi;
      a[u] = reinterpret_cast<uint32_t*>(mc + v * TP_MM);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      if (live[u]) tp_mm_ld(a[u], x[u][0], x[u][1], x[u][2], x[u][3]);
#pragma unroll
    for (int u = 0; u < UNROLL; ++u)
      if (live[u]) tp_mm_st(a[u], x[u][0], x[u][1], x[u][2], x[u][3]);
  }
}

__device__ __forceinline__ float warp_reduce_max_float(float val) {
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1)
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  return val;
}

// ======================== Kernel ========================

// ===== M-tiled N=64 fused compute (Step 8) =====
__device__ static __forceinline__ void wgmma_m64n64k32_e4m3_e4m3_f32(
    uint64_t desc_a, uint64_t desc_b, float* d) {
  constexpr uint32_t scale_D = 1;
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "%32, %33, p, %35, %36;\n}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
        "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),
        "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),
        "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
      : "l"(desc_a),"l"(desc_b),"r"(scale_D),"n"(1),"n"(1));
}

static constexpr int N_FEAT = 64;
static constexpr int N_FEAT_DOWN = H_DIM / N_FEAT;
static constexpr int NUM_WG = 2;
// ===== Tile height and the A pipeline =====
//
// The tile height used to be capped at 48 by a full-K activation cache:
// ACT_CACHE = N_KBLK * A_ROWS * 128 held the WHOLE K extent of the tile, 149504
// of 223552 B at 48 rows, so 48 was the last multiple of 8 that fit and 56 was
// already over the 232448 B SM90 cap. That cap is what set the tile count, the
// tile count is what set the weight DRAM bytes (a tile reads all 7.078 MB of one
// expert's sharded fp8 weights), and the weight bytes are what the kernel is
// bound by. MEASURED: 4.25 TB/s achievable on this H200, and at 12717 tokens the
// deficit against Triton decomposes as 1.298x the bytes times 1.078x worse
// bandwidth efficiency.
//
// DeepGEMM's SM90 fp8 kernel works from the identical 232448 B budget and does
// not hold a full-K cache: smem_a_per_stage = block_m * block_k, num_stages >= 3,
// and its block_m candidates are {64, 128}. This build follows that shape. A is a
// ring of A_STAGES single-k-block slots fed by cp.async, so the SHM cost of A
// stops scaling with K and starts scaling with the pipeline depth, which frees
// the height to reach 128.
//
// The WGMMA is m64n64k32, so a tile taller than 64 rows is issued as M_SUBS
// sub-tiles of 64 rows against the SAME resident weight tile. That is the whole
// point: the weight tile is fetched once and amortized over M_SUBS times as many
// rows, which is the byte win. It costs one extra accumulator set per sub-tile
// (32 registers) and no extra weight traffic.
//
// The price is that A is re-read once per (gate, up) pass pair instead of once
// per tile: N_HALF/N_FEAT = 12 re-reads. That is evidenced to be affordable
// rather than assumed -- Triton's own W+A traffic already implies 4.35/5.02/5.34
// TB/s at 3194/6370/12717 tokens against the 4.25 TB/s ceiling, so the arm we
// lose to is already having its A re-reads served from cache.
static constexpr int A_ROWS = 128;                   // tile height (the knob)
static constexpr int M_SUBS = (A_ROWS + 63) / 64;     // m64 WGMMA sub-tiles
static constexpr int A_PAD = M_SUBS * 64;             // rows the WGMMA reads
static constexpr int A_BLK = A_PAD * 128;             // one k-block of A
static constexpr int A_STAGES = 4;                    // A ring depth (>=3, DeepGEMM's floor)
static constexpr int A_RING = A_STAGES * A_BLK;
static constexpr int B_SHM = 64 * 128;
static constexpr int UP_PIPE = 4;
static constexpr int DN_PIPE = 2;
static constexpr uint64_t A_LBO = 16ULL, A_SBO = 1024ULL; static constexpr uint32_t A_SWZ = 1u;
static constexpr int WGMMAS_PER_128K = 4;
static constexpr int SCALE_GROUPS = D_KBLK;
static constexpr int INTER_SHM = D_KBLK * A_BLK;
static constexpr int SCALE_SHM_B = A_PAD * SCALE_GROUPS * 4;
// Per-row (flattened topk index, routing weight), resolved once per tile. These
// live in PERSIST rather than the transient union because BOTH phases read them:
// the up phase needs the token id to gather A, the down phase needs the same
// index and its weight to scatter the result. Keeping them here also removes the
// sorted_ids GMEM re-read the staging loop used to do per row per k-block.
static constexpr int ROW_META_B = A_PAD * 4 + A_PAD * 4;
static constexpr int PERSIST_RAW = INTER_SHM + SCALE_SHM_B + ROW_META_B;
// Both WGMMA operands start at trans = shm + PERSIST and are described with
// swizzle mode 1, whose e4m3 atom is 8 rows by 128 B = 1024 B, so that base has
// to be 1024 B aligned or the hardware XOR pattern sits offset from the one the
// staging code wrote by hand and the operand is assembled from the wrong bytes.
static constexpr int PERSIST = ((PERSIST_RAW + 1023) / 1024) * 1024;
static constexpr int UP_TRANS = A_RING + NUM_WG*UP_PIPE*B_SHM + NUM_WG*UP_PIPE*16;
static constexpr int DN_TRANS = NUM_WG*DN_PIPE*B_SHM + NUM_WG*DN_PIPE*16;  // direct-read: no a_base
static constexpr int WB_STAGE_B = NUM_WG*64*64*4;
static constexpr int DN_TOTAL = DN_TRANS + WB_STAGE_B;
static constexpr int TRANS = (UP_TRANS > DN_TOTAL) ? UP_TRANS : DN_TOTAL;
static constexpr int SHM_TOTAL = PERSIST + TRANS + 256;
static_assert(SHM_TOTAL <= 227*1024, "fused SHM exceeds cap");
static constexpr int BLOCK_M = A_ROWS;
// A tile may never be taller than the staged rows, or the WGMMA would be
// fed the next block's data as though it belonged to this tile.
static_assert(BLOCK_M <= A_ROWS, "tile height exceeds staged rows");
// The swizzle repeats every 8 rows, so a stride that is not a multiple of
// 8 would shift the swizzle phase at each block boundary. 64 rows per sub-tile
// is 8 whole atoms, which is what lets sub-tile s start at s*64*128 with the
// swizzle phase of row 0 and no re-layout.
static_assert(A_ROWS % 8 == 0, "staged rows must be a multiple of 8");
static_assert(A_BLK % 1024 == 0, "A ring slot must be swizzle-atom aligned");
// >=3 is DeepGEMM's own floor for the same reason it applies here: two stages
// leaves no slot to fill while the WGMMA drains the current one.
static_assert(A_STAGES >= 3, "A ring needs at least three stages");
static_assert(A_STAGES <= N_KBLK, "A ring deeper than the k extent it walks");

// ===== A staging primitives =====
// 16 B is the widest cp.async and it is exactly the swizzle atom's chunk, so a
// contiguous 16 B GMEM chunk lands as a contiguous 16 B SHM chunk -- identical
// bytes to the synchronous staging loop this replaces, which is what makes the
// change bit-checkable rather than merely plausible.
__device__ static __forceinline__ void cp_async16(void* smem_dst,
                                                  const void* gmem_src) {
  const uint32_t d = cvta_to_shared_u32(smem_dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               ::"r"(d), "l"(gmem_src) : "memory");
}
__device__ static __forceinline__ void cpa_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ static __forceinline__ void cpa_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}


// ===== P2b-lite: per-warpgroup named barrier =====
// bar.sync with a non-zero id so it never aliases __syncthreads() (barrier 0).
// The "memory" clobber stops the compiler sinking shared-memory accesses across it.
__device__ __forceinline__ void wg_bar_sync(int id, int count) {
  asm volatile("barrier.sync %0, %1;" :: "r"(id), "r"(count) : "memory");
}

__device__ void up_phase(
    uint8_t* fp8_inter, float* inter_scales, int* row_fi, float* row_rw,
    char* trans,
    const uint8_t* act_fp8, const float* act_scales,
    const float* up_w_scales, const int32_t* sorted_ids,
    const float* topk_weights,
    int e, int row0, int rows, const CUtensorMap& up_w_tma_desc)
{
    const int PIPE_DEPTH = UP_PIPE;
    uint8_t* a_ring = (uint8_t*)trans;
    uint8_t* b_base = a_ring + A_RING;
    uint64_t* bar = (uint64_t*)(b_base + NUM_WG*PIPE_DEPTH*B_SHM);
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32, wg=warp/4, warp_in_wg=warp%4;
    if (tid==0){ for(int s=0;s<NUM_WG*PIPE_DEPTH;++s) mbarrier_init((uint64_t*)((char*)bar+s*16),1u); }
    // Resolve the tile's rows ONCE. Every later use -- the A gather, the
    // per-row activation scale, the down phase's scatter target and its routing
    // weight -- reads these instead of chasing sorted_ids again.
    for (int m = tid; m < A_PAD; m += BLOCK) {
      const int fi = (m < rows) ? sorted_ids[row0 + m] : -1;
      row_fi[m] = fi;
      row_rw[m] = (fi >= 0) ? topk_weights[fi] : 0.f;
    }
    __syncthreads();
    uint32_t parity[UP_PIPE]; for(int s=0;s<PIPE_DEPTH;++s) parity[s]=0;
    uint8_t* b_ring = b_base + wg*PIPE_DEPTH*B_SHM;
    uint64_t* wg_bar = (uint64_t*)((char*)bar + wg*PIPE_DEPTH*16);
        // Stage ONE 128-column k-block of the tile's rows into an A-ring slot.
        //
        // The layout inside a slot is byte-for-byte the layout the full-K cache
        // used for one k-block: 128 B per row, rows contiguous, 16 B chunks
        // XOR-swizzled by (chunk ^ (row & 7)). That is deliberate -- it means the
        // WGMMA descriptor below is unchanged, and it is why replacing the cache
        // with a ring is expected to be BIT-IDENTICAL rather than merely close.
        //
        // Padding rows (m >= rows, and the m64 tail when A_ROWS < A_PAD) are
        // zeroed rather than copied, so a short tile can never feed the WGMMA
        // stale bytes. cp.async has no zero-fill source, so those go through a
        // direct store; both are published by the __syncthreads() that follows
        // the group wait, so the mix is safe.
        // The iteration count is a compile-time constant, not `idx < CH` against a
        // thread-dependent start. That matters: every thread must commit the SAME
        // number of cp.async instructions per group, or `wait_group<N>` means a
        // different thing in different threads and the ring loses its ordering.
        constexpr int A_CH = A_PAD * 8;        // 16 B chunks in one padded k-block
        static_assert(A_CH % BLOCK == 0, "A staging must divide evenly across the block");
        constexpr int A_ITERS = A_CH / BLOCK;
        auto stage_A = [&](int ks, int slot) {
            uint8_t* dst0 = a_ring + slot * A_BLK;
            const uint8_t* src0 = act_fp8 + (size_t)ks * 128;
            #pragma unroll
            for (int it = 0; it < A_ITERS; ++it) {
                const int idx = it * BLOCK + tid;
                const int m = idx >> 3, c = idx & 7;
                uint8_t* d = dst0 + m*128 + ((c ^ (m & 7)) << 4);
                const int fi = row_fi[m];
                if (fi >= 0)
                    cp_async16(d, src0 + (size_t)(fi / TOP_K) * K_DIM + c*16);
                else
                    *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
            cpa_commit();
        };
        // Zero the intermediate scale buffer (for atomicMax accumulation)
        for (int i = tid; i < A_PAD * SCALE_GROUPS; i += BLOCK) inter_scales[i] = 0.0f;
        // Direct-read: zero the whole swizzled intermediate so padding rows are 0.
        // 16 B at a time: INTER_SHM = 32768 scalar byte stores was 128 iterations
        // per thread; as uint4 it is 8. INTER_SHM = 64*N_HALF is a multiple of 16.
        {
            static_assert(INTER_SHM % 16 == 0, "inter zero-fill must be 16 B-divisible");
            uint4* iz = reinterpret_cast<uint4*>(fp8_inter);
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            #pragma unroll
            for (int i = tid; i < INTER_SHM/16; i += BLOCK) iz[i] = z;
        }
        __syncthreads();

        // Paired-pass fused SiLU: GATE_PASSES gate passes paired with the same
        // number of up passes. WG w does gate-pass g = w, w+NUM_WG, ...
        // For paired pass, compute gate acc + up acc, SiLU inline, write FP8 intermediate.
        // The counts below are DERIVED, not the 8/4/128 the old comments claimed:
        // at N_HALF=768 and N_FEAT=64 there are 12 gate passes, 6 per warpgroup,
        // and with K_DIM=3072 (N_KBLK=24) that is 288 tiles per warpgroup.
        const int GATE_PASSES = N_HALF / N_FEAT;
        const int PP_PER_WG = GATE_PASSES / NUM_WG;
        // A "tile" = one (feature-pass, ks). Each paired pass has 2*N_KBLK tiles
        // (gate ks 0..N_KBLK-1, then up ks 0..N_KBLK-1).
        const int WG_TILES = PP_PER_WG * 2 * N_KBLK;
        // Map a WG-local linear tile index -> (global feature-pass gfp, ks).
        auto tile_gfp_ks = [&](int lin, int& gfp, int& ks) {
            int pp = lin / (2 * N_KBLK);          // which paired pass (0..3)
            int within = lin % (2 * N_KBLK);      // 0..31
            int is_up = within / N_KBLK;          // 0=gate, 1=up
            ks = within % N_KBLK;
            int g = wg + pp * NUM_WG;             // gate feature-pass 0..7
            gfp = is_up ? (GATE_PASSES + g) : g;  // up passes are 8..15
        };
        auto arm_tile = [&](int lin, int slot) {
            int gfp, ks; tile_gfp_ks(lin, gfp, ks);
            uint64_t* b = (uint64_t*)((char*)wg_bar + slot*16);
            mbarrier_arrive_expect_tx(b, B_SHM);
            tma_load_2d(up_w_tma_desc, (uint32_t)(ks * 128),
                        (uint32_t)(e * N_UP + gfp * N_FEAT),
                        b_ring + slot*B_SHM, b);
        };
        bool wg_leader = (warp_in_wg == 0 && lane == 0);
        if (wg_leader) {
            #pragma unroll
            for (int s = 0; s < PIPE_DEPTH; ++s)
                if (s < WG_TILES) arm_tile(s, s);
        }

        // A-ring prologue: A_STAGES-1 groups in flight before the first consume.
        // The k-block a warpgroup needs at step `lin` is lin % N_KBLK, and BOTH
        // warpgroups walk the same `lin`, so ONE shared ring serves both. That is
        // the whole point: A is fetched once per k-step, not once per warpgroup,
        // so dropping the full-K cache costs a re-read per feature-pass pair and
        // nothing more.
        #pragma unroll
        for (int s = 0; s < A_STAGES-1; ++s) stage_A(s % N_KBLK, s % A_STAGES);

        // Row within the m64 sub-tile. The absolute row is s*64 + m0b for
        // sub-tile s, which is what every `< rows` test below uses.
        const int m0b = warp_in_wg*16 + lane/4;
        const int m1b = m0b + 8;

        // P2c: hoist the sorted_ids load + divide out of the k-loop. The token
        // index for a given m does not change across pp/half/ks, but the loop
        // was re-issuing the dependent GMEM load once per tile-step. The flat id
        // now comes from row_fi (SHM, resolved once per tile) rather than from a
        // second GMEM chase.
        const float* as_p[M_SUBS][2];
        #pragma unroll
        for (int s = 0; s < M_SUBS; ++s) {
            const int f0 = row_fi[s*64 + m0b];
            const int f1 = row_fi[s*64 + m1b];
            as_p[s][0] = (f0 >= 0) ? (act_scales + (size_t)(f0/TOP_K)*N_KBLK) : nullptr;
            as_p[s][1] = (f1 >= 0) ? (act_scales + (size_t)(f1/TOP_K)*N_KBLK) : nullptr;
        }

        for (int pp = 0; pp < PP_PER_WG; ++pp) {
            int g = wg + pp * NUM_WG;            // gate feature-pass
            float acc_g[M_SUBS][32], acc_u[M_SUBS][32];
            #pragma unroll
            for (int s = 0; s < M_SUBS; ++s) {
                #pragma unroll
                for (int i = 0; i < 32; ++i) { acc_g[s][i]=0.f; acc_u[s][i]=0.f; }
            }

            // half=0: gate into acc_g; half=1: up into acc_u
            for (int half = 0; half < 2; ++half) {
                int gfp = half ? (GATE_PASSES + g) : g;
                int feat_base = gfp * N_FEAT;
                for (int ks = 0; ks < N_KBLK; ++ks) {
                    int lin = pp * 2 * N_KBLK + half * N_KBLK + ks;

                    // A[lin] has landed. wait_group<A_STAGES-2> leaves at most
                    // A_STAGES-2 groups pending, and groups retire in issue order,
                    // so the group that staged this k-block is complete. The refill
                    // is issued UNCONDITIONALLY, including past the end of the tile:
                    // a conditional refill would shrink the pending count in the last
                    // two steps, at which point wait_group<A_STAGES-2> is satisfied
                    // without the step's own group having landed. Two wasted stagings
                    // out of WG_TILES is the price of a wait that always means what it
                    // says.
                    cpa_wait<A_STAGES-2>();
                    __syncthreads();
                    // Safe to overwrite slot (lin-1) % A_STAGES: the previous step
                    // ended in wgmma_wait_group<0> in BOTH warpgroups before either
                    // reached the __syncthreads() just above, so no WGMMA is still
                    // reading it.
                    stage_A((lin + A_STAGES-1) % N_KBLK, (lin + A_STAGES-1) % A_STAGES);

                    int slot = lin % PIPE_DEPTH;
                    uint64_t* cur_bar = (uint64_t*)((char*)wg_bar + slot*16);
                    while (!mbarrier_try_wait_parity(cur_bar, parity[slot])) {}
                    parity[slot] ^= 1u;
                    uint8_t* a_slot = a_ring + (lin % A_STAGES) * A_BLK;
                    uint8_t* b_cur = b_ring + slot*B_SHM;

                    float ws = up_w_scales[(size_t)e*(N_UP/128)*N_KBLK + (size_t)(feat_base/128)*N_KBLK + ks];

                    // A_ROWS rows are issued as M_SUBS sub-tiles of 64 against the
                    // SAME resident weight tile b_cur. That is what buys the byte
                    // amortization: the weight bytes are read once and divided over
                    // M_SUBS times as many rows. The instruction shape is m64n64k32,
                    // so the sub-tiling is not optional at A_ROWS > 64 -- the only
                    // cost is one more accumulator pair per sub-tile.
                    #pragma unroll
                    for (int sb = 0; sb < M_SUBS; ++sb) {
                        uint8_t* a_ptr = a_slot + sb * (64*128);
                        float d[32];
                        #pragma unroll
                        for (int i = 0; i < 32; ++i) d[i] = 0.f;
                        wgmma_fence();
                        for (int j = 0; j < WGMMAS_PER_128K; ++j) {
                            uint64_t da = make_wgmma_desc(a_ptr + j*32u, A_LBO, A_SBO, A_SWZ);
                            uint64_t db = make_wgmma_desc(b_cur + j*32u, 16ULL, 1024ULL, 1u);
                            wgmma_m64n64k32_e4m3_e4m3_f32(da, db, d);
                        }
                        wgmma_commit_group();
                        wgmma_wait_group<0>();

                        float as0 = as_p[sb][0] ? as_p[sb][0][ks] : 0.f;  // P2c: chase hoisted
                        float as1 = as_p[sb][1] ? as_p[sb][1][ks] : 0.f;  // P2c: chase hoisted
                        float s0 = ws*as0, s1 = ws*as1;
                        if (half == 0) {
                            #pragma unroll
                            for (int dr = 0; dr < 32; ++dr)
                                acc_g[sb][dr] += d[dr] * (((dr/2)%2==0) ? s0 : s1);
                        } else {
                            #pragma unroll
                            for (int dr = 0; dr < 32; ++dr)
                                acc_u[sb][dr] += d[dr] * (((dr/2)%2==0) ? s0 : s1);
                        }
                    }
                    // P2e: barrier REMOVED. wgmma.wait_group.sync.aligned tracks a group
                    // counter scoped to the WARPGROUP, not to the individual warp, and
                    // wgmma.mma_async is warpgroup-collective. So once wait_group<0> is
                    // satisfied, every WGMMA of this warpgroup has already finished
                    // reading b_ring -- which is the only thing this barrier protected
                    // (it kept wg_leader from re-arming a slot under a lagging warp).
                    // Redundant, not load-bearing. Gated on BITWISE equality to fp16s2
                    // plus 8/8 run-to-run bitwise self-identity, NOT on cosine
                    // similarity: cos = 1.0000000 passed a genuinely racy variant
                    // earlier in this campaign, so cos cannot gate a race.
                    // The re-arm moved BELOW the sub-tile loop: every sub-tile reads
                    // the same b_cur, so the slot is only free once the last one has
                    // drained.
                    if (wg_leader && lin + PIPE_DEPTH < WG_TILES)
                        arm_tile(lin + PIPE_DEPTH, slot);
                }
            }

            // SiLU(gate)*up per dreg, then per-lane group max for the FP8 scale.
            // The intermediate column for dreg dr = g*N_FEAT + dfrag_col.
            // Each lane holds 32 dregs spanning 2 M-rows x 16 N-cols of this g-block,
            // per sub-tile. The product is written back over acc_u rather than into a
            // third array: at M_SUBS=2 a separate sv[M_SUBS][32] would add 64 live
            // registers on top of the 160 already needed, and 224 is where the
            // occupancy-1 register budget starts to bind.
            #pragma unroll
            for (int sb = 0; sb < M_SUBS; ++sb) {
                #pragma unroll
                for (int dr = 0; dr < 32; ++dr) {
                    float gg = acc_g[sb][dr];
                    acc_u[sb][dr] = __fdividef(acc_u[sb][dr] * gg, 1.0f + __expf(-gg));
                }
            }
            // Per-(row, 128-feat block) scale. This g-block covers feat [g*64, g*64+64),
            // all within 128-block (g*64)/128. Two consecutive g share a 128-block, so
            // the intermediate scale group is (g*64)/128 = g/2.
            int sg = (g * N_FEAT) / 128;  // scale-group index in [0, SCALE_GROUPS)
            // Warp-reduce across lanes that share the same (row, sg): all 32 lanes of
            // the warp contribute to the same 64-col block for their 2 rows. But rows
            // differ by lane/4. Use atomicMax into the SHM scale buffer per (row, sg).
            // EVERY sub-tile's contribution must land before ANY thread reads the
            // reduced value back, which is why the atomicMax sweep and the quantize
            // sweep are two separate loops over sb with a barrier between them.
            #pragma unroll
            for (int sb = 0; sb < M_SUBS; ++sb) {
                float mx0 = 0.f, mx1 = 0.f;
                #pragma unroll
                for (int dr = 0; dr < 32; ++dr) {
                    float a = fabsf(acc_u[sb][dr]);
                    if ((dr/2)%2==0) mx0 = fmaxf(mx0, a); else mx1 = fmaxf(mx1, a);
                }
                const int mr0 = sb*64 + m0b, mr1 = sb*64 + m1b;
                if (mr0 < rows)
                    atomicMax(reinterpret_cast<int*>(&inter_scales[mr0*SCALE_GROUPS + sg]), __float_as_int(mx0));
                if (mr1 < rows)
                    atomicMax(reinterpret_cast<int*>(&inter_scales[mr1*SCALE_GROUPS + sg]), __float_as_int(mx1));
            }
            __syncthreads();
            // Now inter_scales holds the abs-max; convert to scale and quantize.
            #pragma unroll
            for (int sb = 0; sb < M_SUBS; ++sb) {
                const int mr0 = sb*64 + m0b, mr1 = sb*64 + m1b;
                float sc0 = (mr0<rows) ? inter_scales[mr0*SCALE_GROUPS + sg] : 1.f;
                float sc1 = (mr1<rows) ? inter_scales[mr1*SCALE_GROUPS + sg] : 1.f;
                // Guard: the max may be 0 for padding.
                float inv0 = (sc0 > __FLT_MIN__) ? (448.0f / sc0) : 0.f;
                float inv1 = (sc1 > __FLT_MIN__) ? (448.0f / sc1) : 0.f;
                #pragma unroll
                for (int dr = 0; dr < 32; ++dr) {
                    int m = mr0 + ((dr/2)%2)*8;
                    int col = g*N_FEAT + (lane%4)*2 + dr%2 + (dr/4)*8;
                    if (m < rows) {
                        float inv = ((dr/2)%2==0) ? inv0 : inv1;
                        __nv_fp8_e4m3 q = (__nv_fp8_e4m3)(acc_u[sb][dr] * inv);
                        // Direct-read layout: block dk=col/128, swizzled [A_PAD,128].
                        int dk = col / 128, kk = col % 128;
                        int sw = (kk / 16) ^ (m & 7);
                        fp8_inter[dk*A_BLK + m*128 + sw*16 + (kk % 16)] = *reinterpret_cast<uint8_t*>(&q);
                    }
                }
            }
            __syncthreads();
        }
        // Drain the A ring before leaving. Without this, the refills issued past
        // the end of the tile are still in flight when the NEXT tile's prologue
        // writes the same slots, and two cp.async groups targeting one address with
        // no wait between them have no defined order.
        cpa_wait<0>();

        // Convert the stored abs-max scales to actual scales (max/448) for the down-proj.
        for (int i = tid; i < rows * SCALE_GROUPS; i += BLOCK) {
            float mx = inter_scales[i];
            inter_scales[i] = (mx > __FLT_MIN__) ? (mx / 448.0f) : (1.0f/448.0f);
        }
        __syncthreads();


}

__device__ void down_phase(
    uint8_t* fp8_inter, float* inter_scales, const int* row_fi, const float* row_rw,
    char* trans,
    const uint8_t* down_w_fp8, const float* down_w_scales,
    float* output_accum, const float* topk_weights, const int32_t* sorted_ids,
    int e, int row0, int rows, const CUtensorMap& down_w_tma_desc,
    int num_slots)
{
    const int PIPE_DEPTH = DN_PIPE;
    // Direct-read: no a_base (A-operand read in place from fp8_inter).
    uint8_t* b_base = (uint8_t*)trans;
    uint64_t* bar = (uint64_t*)(b_base + NUM_WG*PIPE_DEPTH*B_SHM);
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32, wg=warp/4, warp_in_wg=warp%4;
    if (tid==0){ for(int s=0;s<NUM_WG*PIPE_DEPTH;++s) mbarrier_init((uint64_t*)((char*)bar+s*16),1u); }
    __syncthreads();
    uint32_t parity[DN_PIPE]; for(int s=0;s<PIPE_DEPTH;++s) parity[s]=0;
    // Direct-read: a_stg is set per-dk inside the loop (fp8_inter block).
    uint8_t* b_ring = b_base + wg*PIPE_DEPTH*B_SHM;
    uint64_t* wg_bar = (uint64_t*)((char*)bar + wg*PIPE_DEPTH*16);
    const int PASSES_PER_WG = N_FEAT_DOWN / NUM_WG;
        // ═══════════ DOWN-PROJ (2 WG, each does PASSES_PER_WG passes) ═══════════
        // WG w processes global fp = w, w+NUM_WG, w+2*NUM_WG, ...
        // arm helper: linear index over this WG's (local_pass, dk)
        auto arm = [&](int gfp, int dk, int slot) {
            int feat_base = gfp * N_FEAT;
            uint64_t* b = (uint64_t*)((char*)wg_bar + slot*16);
            mbarrier_arrive_expect_tx(b, B_SHM);
            tma_load_2d(down_w_tma_desc, (uint32_t)(dk * 128),
                        (uint32_t)(e * H_DIM + feat_base),
                        b_ring + slot*B_SHM, b);
        };
        const int WG_TOTAL = PASSES_PER_WG * D_KBLK;  // 64 tiles per WG
        // linear tile index -> (local_pass p, dk); gfp = wg + p*NUM_WG
        // Pre-arm first PIPE_DEPTH tiles (TMA launcher = lane 0 of WG's first warp)
        bool wg_leader = (warp_in_wg == 0 && lane == 0);
        if (wg_leader) {
            #pragma unroll
            for (int s = 0; s < PIPE_DEPTH; ++s) {
                if (s < WG_TOTAL) {
                    int p = s / D_KBLK, dk = s % D_KBLK;
                    arm(wg + p*NUM_WG, dk, s);
                }
            }
        }

        // (fi, rw) per row: the coalesced writeback re-maps which thread owns which
        // row, so a per-thread register copy will not serve it. These used to be
        // re-derived here from sorted_ids into the idle portion of the down-phase
        // SHM union; they now come from the PERSIST-resident row_fi/row_rw that
        // up_phase already resolved for the same tile, which removes a second
        // sorted_ids/topk_weights chase per tile and frees the union slot.

        for (int p = 0; p < PASSES_PER_WG; ++p) {
            int gfp = wg + p * NUM_WG;   // global feature pass
            int feat_base = gfp * N_FEAT;
            float acc[M_SUBS][32];
            #pragma unroll
            for (int sb = 0; sb < M_SUBS; ++sb) {
                #pragma unroll
                for (int i = 0; i < 32; ++i) acc[sb][i] = 0.f;
            }

            for (int dk = 0; dk < D_KBLK; ++dk) {
                int lin = p * D_KBLK + dk;
                int slot = lin % PIPE_DEPTH;

                // Direct-read: A-operand is fp8_inter block dk, already swizzled.
                uint8_t* a_stg = fp8_inter + dk * A_BLK;
                uint64_t* cur_bar = (uint64_t*)((char*)wg_bar + slot*16);
                while (!mbarrier_try_wait_parity(cur_bar, parity[slot])) {}
                parity[slot] ^= 1u;
                uint8_t* b_cur = b_ring + slot*B_SHM;
                // P2e: barrier REMOVED. Each thread already performed its own
                // mbarrier_try_wait_parity on the TMA arrival just above, and up_phase
                // writes fp8_inter before the __syncthreads() that separates the two
                // phases in the tile loop, so those writes are already published. No
                // additional warpgroup rendezvous is implied here. Same bitwise gate.

                // Scale-apply
                int feat_block = feat_base / 128;
                float dws = down_w_scales[(size_t)e*(H_DIM/128)*D_KBLK +
                                          (size_t)feat_block*D_KBLK + dk];
                const int m0b = warp_in_wg*16 + lane/4, m1b = m0b + 8;

                // Same sub-tiling as the up phase, same reason: one resident weight
                // tile serves M_SUBS m64 sub-tiles, so the down-projection weight
                // bytes are amortised over A_ROWS rows instead of 64.
                #pragma unroll
                for (int sb = 0; sb < M_SUBS; ++sb) {
                    float d[32];
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) d[i] = 0.f;
                    wgmma_fence();
                    for (int j = 0; j < WGMMAS_PER_128K; ++j) {
                        uint64_t da = make_wgmma_desc(a_stg + sb*(64*128) + j*32, A_LBO, A_SBO, A_SWZ);
                        uint64_t db = make_wgmma_desc(b_cur + j*32, 16ULL, 1024ULL, 1u);
                        wgmma_m64n64k32_e4m3_e4m3_f32(da, db, d);
                    }
                    wgmma_commit_group();
                    wgmma_wait_group<0>();

                    const int mr0 = sb*64 + m0b, mr1 = sb*64 + m1b;
                    float as0 = (mr0 < rows) ? inter_scales[mr0*SCALE_GROUPS + dk] : 0.0f;
                    float as1 = (mr1 < rows) ? inter_scales[mr1*SCALE_GROUPS + dk] : 0.0f;
                    #pragma unroll
                    for (int dr = 0; dr < 32; ++dr) {
                        float sc = ((dr/2)%2 == 0) ? (dws*as0) : (dws*as1);
                        acc[sb][dr] += d[dr] * sc;
                    }
                }
                // P2e: barrier REMOVED -- same argument as the up_phase note above.
                // Moved below the sub-tile loop: every sub-tile reads the same
                // b_cur, so the slot is only free once the last one has drained.
                if (wg_leader && lin + PIPE_DEPTH < WG_TOTAL) {
                    int np = (lin + PIPE_DEPTH) / D_KBLK, ndk = (lin + PIPE_DEPTH) % D_KBLK;
                    arm(wg + np*NUM_WG, ndk, slot);
                }
                // P2f: barrier REMOVED. The only work between the previous barrier
                // and this point is acc[dr] += d[dr] * sc -- register-local per thread,
                // writing no shared memory, so there is nothing here for a warpgroup
                // rendezvous to publish. This one was found by READING the down_phase
                // loop rather than by pattern-matching the three sites above, and it is
                // worth MORE than those three combined (-2.5% vs -1.7% over fp16s2).
            }

            // Writeback, FEATURE-MAJOR oa: [NPASS][BS*TOP_K][N_FEAT]. MEASURED scatter
            // was 0.0% adjacent rows / median gap 1.7 MiB / ~242 MB span per tile.
            // In this layout one pass's 64 rows are 128 B apart => 8 KiB contiguous.
            // The staging buffer stays 64 rows wide and the writeback runs in
            // M_SUBS rounds. Sizing it to A_ROWS instead would add
            // NUM_WG*64*64*4 = 32 KiB of shared memory per extra sub-tile, and
            // shared memory is the budget that caps the tile height in the first
            // place -- the whole point of this build. The rounds cost one extra
            // warpgroup rendezvous pair each, against a per-tile wall time of
            // ~283 us.
            {
                float* wb_stage = (float*)(trans + DN_TRANS) + wg*(64*64);
                __nv_bfloat16* oa16 = reinterpret_cast<__nv_bfloat16*>(output_accum);
                const size_t plane = (size_t)gfp * ((size_t)num_slots * N_FEAT);
                const int twg = warp_in_wg*32 + lane;
                const int r_in = twg / 8;
                const int c8 = (twg % 8) * 8;
                #pragma unroll
                for (int sb = 0; sb < M_SUBS; ++sb) {
                    #pragma unroll
                    for (int dr = 0; dr < 32; ++dr) {
                        int m = warp_in_wg*16 + lane/4 + ((dr/2)%2)*8;
                        int n = (lane%4)*2 + dr%2 + (dr/4)*8;
                        wb_stage[m*64 + n] = acc[sb][dr];
                    }
                    wg_bar_sync(wg + 1, 128);
                    #pragma unroll
                    for (int rb = 0; rb < 64; rb += 16) {
                        int m = sb*64 + rb + r_in;
                        if (m < rows) {
                            int fi = row_fi[m];
                            // No scale to fold in: bf16 covers the fp32 exponent
                            // range, so the partial is stored as it stands.
                            float rw = row_rw[m];
                            __nv_bfloat16 tmp[8];
                            #pragma unroll
                            for (int q = 0; q < 8; ++q) {
                                // Saturate instead of overflowing to Inf. bf16's max
                                // normal is 3.39e38, so this is unreachable for any
                                // measured partial (M2's peak is 1339.5477); it exists
                                // so that an unseen larger partial degrades to a
                                // bounded value rather than an Inf that poisons the
                                // next layer's router.
                                float v = wb_stage[(rb + r_in)*64 + c8 + q]*rw;
                                v = fminf(fmaxf(v, -OA_BF16_CLAMP), OA_BF16_CLAMP);
                                tmp[q] = __float2bfloat16(v);
                            }
                            *reinterpret_cast<uint4*>(
                                &oa16[plane + (size_t)fi*N_FEAT + c8]) =
                                *reinterpret_cast<uint4*>(tmp);
                        }
                    }
                    wg_bar_sync(wg + 1, 128);
                }
            }
        }

}


__global__ void __launch_bounds__(BLOCK_SIZE, 1) prefill_moe_wgmma_kernel(
    const float* __restrict__ logits,        // [num_tokens, NUM_EXPERTS]
    int num_tokens,
    int block_m,
    Workspace* __restrict__ ws,
    int32_t* __restrict__ topk_ids,          // [num_tokens, TOP_K]
    float* __restrict__ topk_weights,        // [num_tokens, TOP_K]
    int32_t* __restrict__ sorted_token_ids,  // [num_tokens * TOP_K]
    const __nv_bfloat16* __restrict__ act_bf16, // [num_tokens, K_DIM] bf16 input
    uint8_t* __restrict__ act_fp8,           // [num_tokens, K_DIM] e4m3 GMEM scratch
    float* __restrict__ act_scales,          // [num_tokens, N_KBLK] GMEM scratch
    const float* __restrict__ up_w_scales,   // [E, N_WROW, N_KBLK]
    const uint8_t* __restrict__ down_w_fp8,  // [E, H_DIM, N_HALF] e4m3
    const float* __restrict__ down_w_scales, // [E, D_WROW, D_KBLK]
    float* __restrict__ output_accum,        // [num_tokens, H_DIM] fp32
    __nv_bfloat16* __restrict__ bf16_output, // [num_tokens, H_DIM]
    const float* __restrict__ router_bias,   // [NUM_EXPERTS] or nullptr
    __grid_constant__ CUtensorMap const up_w_tma_desc,
    __grid_constant__ CUtensorMap const down_w_tma_desc,
    __nv_bfloat16* tp_mc,        // multicast alias over bf16_output, or nullptr
    uint32_t** tp_pads,          // per-rank barrier window base pointers
    int tp_rank, int tp_world,
    int tp_nchunk,               // release the collective in this many chunks
    int tp_ncomm,                // blocks that stop computing and communicate
    uint32_t tp_seq,             // host-supplied launch number; used only
                                 // when tp_seqctr is null
    uint32_t* tp_seqctr,         // persistent device word outside Workspace;
                                 // the graph-safe sequence source
    // The residual folded into the SAME collective. bf16 [num_tokens, H_DIM],
    // rank-local and PARTIAL, exactly like the routed half is partial, so
    // adding it on every rank before the reduce makes the one multimem pass
    // carry out_scale*routed_total + residual_total. Null keeps the shipped
    // store.
    const __nv_bfloat16* __restrict__ tp_residual,
    // Applied to the routed sum in fp32 BEFORE the bf16 round, which is where
    // a per-rank scale is exact: sum_r (a*x_r) == a * sum_r x_r.
    float out_scale) {
  // One dedicated, fresh (zeroed) counter pair per barrier call site; see
  // the NUM_BARRIER_SITES comment for why slot reuse is unsafe here.
  // grid_barrier bumps the phase it is given; we hand each site its own
  // zero phase so every site uses slot 0 of its own pair exactly once.
  int barrier_site = 0;
  auto sync_grid = [&]() {
    uint32_t site_phase = 0;
    moe_monokernel::grid_barrier<GRID_SIZE>(ws->barrier[barrier_site],
                                            site_phase);
    ++barrier_site;
  };

  const int numel = num_tokens * TOP_K;
  const int gtid = static_cast<int>(blockIdx.x * BLOCK_SIZE + threadIdx.x);
  const int gstride = static_cast<int>(GRID_SIZE * BLOCK_SIZE);

  // ------------------------------------------------------------------
  // Graph-safe launch sequence
  // ------------------------------------------------------------------
  // A CUDA graph bakes kernel arguments, so a replay reuses tp_seq while the
  // barrier pads still hold that value; every cross-rank barrier then releases
  // on arrival instead of waiting and the reduce runs before the peers have
  // written. MEASURED: replays 1..3 of a captured launch returned up to 12.8 M
  // wrong bf16 with rc == 0, and reusing a seq with no graph at all reproduces
  // it, so the cause is seq reuse and not capture.
  //
  // tp_seqctr is caller memory OUTSIDE Workspace, because the launcher memsets
  // Workspace every launch and a graph replays that memset, which would reset a
  // counter kept there. One thread bumps it and publishes the new value into
  // Workspace, which the memset has just zeroed, so no block can read a stale
  // sequence. sync_grid() #1 below then makes it visible to every block for
  // free: grid_barrier gives full happens-before, so nothing spins for it and no
  // barrier is added.
  //
  // Forward gaps are harmless because the barrier spin is `v >= seq` (MEASURED:
  // jumps to 1000/1001/5000 are bitwise correct), so the atomicAdd needs no
  // ordering with the peers. Ranks agree on nothing but the count: each rank
  // owns its counter, so the invariant is only that every rank runs the same
  // number of fused launches, which is what the host counter also required.
  //
  // Limit: uint32 wraps after 2^32 launches. At the 2.99 ms MEASURED for 8192
  // tokens that is 149 days of back-to-back fused prefill; on wrap the spin
  // would false-release exactly as reuse does.
  if (tp_seqctr != nullptr && tp_mc != nullptr && tp_pads != nullptr &&
      tp_ncomm > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
    const uint32_t s = atomicAdd(tp_seqctr, 1u) + 1u;
    atomicExch(&ws->tp_seq_pub, s);
  }


  // ------------------------------------------------------------------
  // Phase 0: zero the histogram + softmax-topk routing (1 warp / token)
  // ------------------------------------------------------------------
  // (Phase 1a fused) expert_counts is zeroed by the launcher's cudaMemsetAsync;
  // Phase 0 accumulates the histogram in-place via atomicAdd at the topk write.

  {
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int gwarp = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int nwarps = static_cast<int>(GRID_SIZE) * WARPS_PER_BLOCK;

    for (int t = gwarp; t < num_tokens; t += nwarps) {
      const float* row = logits + static_cast<size_t>(t) * NUM_EXPERTS;

      // Coalesced load: lane owns experts {lane + 32*j}.
      float v[LOGITS_PER_LANE];
#pragma unroll
      for (int j = 0; j < LOGITS_PER_LANE; ++j) {
        v[j] = row[lane + 32 * j];
      }

      // Selection score and written weight are two quantities. Under softmax
      // they coincide, because softmax is monotone in its argument so the argmax
      // over logits is the argmax over probabilities. Under sigmoid with a
      // correction bias they do not: selection uses score + bias and the weight
      // is the unbiased score, so sel[] and unb[] are kept apart.
      float sel[LOGITS_PER_LANE], unb[LOGITS_PER_LANE];
      float m = 0.0f, s = 1.0f;
      if constexpr (ROUTER_MODE == 0) {
        // Row max (for a numerically stable softmax).
        m = v[0];
#pragma unroll
        for (int j = 1; j < LOGITS_PER_LANE; ++j) m = fmaxf(m, v[j]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
        }

        // Softmax denominator.
        s = 0.0f;
#pragma unroll
        for (int j = 0; j < LOGITS_PER_LANE; ++j) s += expf(v[j] - m);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
        }
#pragma unroll
        for (int j = 0; j < LOGITS_PER_LANE; ++j) { sel[j] = v[j]; unb[j] = v[j]; }
      } else {
        // Sigmoid scores. The bias shifts selection only, so the weight keeps
        // the unbiased score, which is what the reference router gathers.
#pragma unroll
        for (int j = 0; j < LOGITS_PER_LANE; ++j) {
          const float sg = 1.0f / (1.0f + expf(-v[j]));
          unb[j] = sg;
          sel[j] = sg + ((router_bias != nullptr) ? router_bias[lane + 32 * j] : 0.0f);
        }
        if constexpr (ROUTER_MODE == 2) {
          // One group per register slot, so a group score is one warp reduction.
          // Group score is the sum of the group's top 2 biased scores. The top 1
          // carries its owning lane so the top 2 excludes exactly that lane and
          // stays a distinct expert even when two experts score identically.
          float gs[LOGITS_PER_LANE];
#pragma unroll
          for (int j = 0; j < LOGITS_PER_LANE; ++j) {
            float b1 = sel[j];
            int i1 = lane;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
              const float ov = __shfl_xor_sync(0xFFFFFFFFu, b1, off);
              const int oi = __shfl_xor_sync(0xFFFFFFFFu, i1, off);
              if (ov > b1 || (ov == b1 && oi < i1)) { b1 = ov; i1 = oi; }
            }
            float b2 = (lane == i1) ? -__FLT_MAX__ : sel[j];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
              b2 = fmaxf(b2, __shfl_xor_sync(0xFFFFFFFFu, b2, off));
            }
            gs[j] = b1 + b2;
          }
          // Every lane now holds every group score, so the group selection is
          // warp uniform and needs no further communication.
          uint32_t gkeep = 0u;
#pragma unroll
          for (int r = 0; r < TOPK_GROUP; ++r) {
            float gb = -__FLT_MAX__;
            int gi = -1;
#pragma unroll
            for (int j = 0; j < LOGITS_PER_LANE; ++j) {
              if (!((gkeep >> j) & 1u) && gs[j] > gb) { gb = gs[j]; gi = j; }
            }
            if (gi >= 0) gkeep |= 1u << gi;
          }
#pragma unroll
          for (int j = 0; j < LOGITS_PER_LANE; ++j) {
            if (!((gkeep >> j) & 1u)) sel[j] = -__FLT_MAX__;
          }
        }
      }

      // Iterative warp-wide argmax, TOP_K rounds.  Ties break toward the
      // lower expert index for determinism.
      uint32_t used = 0;  // per-lane bitmask over sel[j]
      // The renorm denominator, accumulated as the weights are produced. Every
      // lane accumulates the same value: `best` is warp uniform after the shuffle
      // reduction below, and the mode != 0 payload `bunb` rides that same
      // reduction, so the summed quantity is identical in all 32 lanes and no
      // extra communication is needed to agree on it.
      float wsum = 0.0f;
      for (int k = 0; k < TOP_K; ++k) {
        float best = -__FLT_MAX__;
        float bunb = 0.0f;
        int bj = -1;
#pragma unroll
        for (int j = 0; j < LOGITS_PER_LANE; ++j) {
          if (!((used >> j) & 1u) && sel[j] > best) {
            best = sel[j];
            bunb = unb[j];
            bj = j;
          }
        }
        int bidx = (bj < 0) ? INT32_MAX : (lane + 32 * bj);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, off);
          const int oi = __shfl_xor_sync(0xFFFFFFFFu, bidx, off);
          // The unbiased score rides along as a payload. Fetching it after the
          // reduction would need a dynamic index into a per-thread array, which
          // would push that array to local memory. Mode 0 reconstructs its
          // weight from best instead, so this third shuffle compiles out there.
          float ou = 0.0f;
          if constexpr (ROUTER_MODE != 0) {
            ou = __shfl_xor_sync(0xFFFFFFFFu, bunb, off);
          }
          if (ov > best || (ov == best && oi < bidx)) {
            best = ov;
            bidx = oi;
            bunb = ou;
          }
        }
        // All lanes now agree on (best, bidx).  Owner retires its slot.
        if (lane == (bidx & 31)) {
          used |= 1u << (bidx >> 5);
        }
        // Computed once, by every lane, so that the value summed into wsum and the
        // value stored to topk_weights are the same expression and cannot diverge.
        const float wk = (ROUTER_MODE == 0) ? (expf(best - m) / s) : bunb;
        wsum += wk;
        if (lane == 0) {
          topk_ids[t * TOP_K + k] = bidx;
          topk_weights[t * TOP_K + k] = wk;
          atomicAdd(&ws->expert_counts[bidx], 1);  // Phase 1a fused into routing
        }
      }
      // FUSED RENORM. The scaled quantity is exactly what the caller used to sum,
      // including the degenerate slot where no expert remained and bidx came back
      // INT32_MAX, so the denominator matches topk_weights.sum(1) term for term
      // rather than merely in the normal case. Cost is 8 fp32 round trips through
      // L1 per token against the H_DIM bf16 read-modify-write this replaces.
      if (lane == 0) {
        const float inv = 1.0f / fmaxf(wsum, 1e-30f);  // matches clamp(min=1e-30)
#pragma unroll
        for (int kk = 0; kk < TOP_K; ++kk) topk_weights[t * TOP_K + kk] *= inv;
      }
    }
  }

  // ------------------------------------------------------------------
  // Phase 0.5: quantize each ORIGINAL token ONCE (BF16 -> FP8 + per-128-K scale)
  // into the act_fp8/act_scales GMEM scratch. Runs concurrently with Phase 0
  // (no dependence on routing); sync #1 separates these writes from Phase 2's
  // read. Loop over num_tokens (NOT the gathered num_tokens*top_k) => quantize
  // once, not 8x. One warp per (token, ksb); 32 lanes x 4 K = 128.
  {
    const int gwarp = static_cast<int>(blockIdx.x) * WARPS_PER_BLOCK + (static_cast<int>(threadIdx.x) / 32);
    const int nwarps_grid = static_cast<int>(GRID_SIZE) * WARPS_PER_BLOCK;
    const int qlane = static_cast<int>(threadIdx.x) % 32;
    const long long nblk = (long long)num_tokens * N_KBLK;
    // MLP-4: 4 INDEPENDENT (token, ksb) pairs per warp iteration. The
    // access width was already fixed (8 B load / 4 B store) and bought only
    // -0.23%, because one 8 B load per lane leaves just 1024 warps x 32 lanes x
    // 8 B = 256 KB in flight against the ~4.8 MB needed to cover HBM latency.
    // Issuing 4 provably independent loads (distinct blk) before the first
    // consumer lets them overlap. Arithmetic per pair is untouched and pairs
    // never interact, so the result is BITWISE identical.
    constexpr int MLPQ = 4;
    const long long stride_q = (long long)nwarps_grid * MLPQ;
    const int c = qlane * 4;
    long long blk0 = gwarp;
    for (; blk0 + (long long)nwarps_grid * (MLPQ - 1) < nblk; blk0 += stride_q) {
      long long blk[MLPQ];
      uint2 braw[MLPQ];
      #pragma unroll
      for (int u = 0; u < MLPQ; ++u) {
        blk[u] = blk0 + (long long)nwarps_grid * u;
        const int tok = (int)(blk[u] / N_KBLK);
        const int ksb = (int)(blk[u] % N_KBLK);
        braw[u] = *reinterpret_cast<const uint2*>(
            act_bf16 + (size_t)tok * K_DIM + (size_t)ksb * 128 + c);
      }
      #pragma unroll
      for (int u = 0; u < MLPQ; ++u) {
        const int tok = (int)(blk[u] / N_KBLK);
        const int ksb = (int)(blk[u] % N_KBLK);
        const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&braw[u]);
        float r0 = __bfloat162float(bp[0]);
        float r1 = __bfloat162float(bp[1]);
        float r2 = __bfloat162float(bp[2]);
        float r3 = __bfloat162float(bp[3]);
        float lm = fmaxf(fmaxf(fabsf(r0),fabsf(r1)), fmaxf(fabsf(r2),fabsf(r3)));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          lm = fmaxf(lm, __shfl_xor_sync(0xFFFFFFFFu, lm, off));
        if (lm < __FLT_MIN__) lm = 1.f;
        const float inv = 448.0f / lm;
        uint32_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          float rv = (j==0)?r0:(j==1)?r1:(j==2)?r2:r3;
          __nv_fp8_e4m3 qv = (__nv_fp8_e4m3)(rv * inv);
          packed |= (uint32_t)(*reinterpret_cast<uint8_t*>(&qv)) << (8*j);
        }
        *reinterpret_cast<uint32_t*>(act_fp8 + (size_t)tok * K_DIM
                                     + (size_t)ksb * 128 + c) = packed;
        if (qlane == 0) act_scales[(size_t)tok * N_KBLK + ksb] = lm / 448.0f;
      }
    }
    // Remainder: fewer than MLPQ pairs left for this warp -- same body, scalar.
    for (long long blk = blk0; blk < nblk; blk += nwarps_grid) {
      const int tok = (int)(blk / N_KBLK);
      const int ksb = (int)(blk % N_KBLK);
      const uint2 braw = *reinterpret_cast<const uint2*>(
          act_bf16 + (size_t)tok * K_DIM + (size_t)ksb * 128 + c);
      const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(&braw);
      float r0 = __bfloat162float(bp[0]);
      float r1 = __bfloat162float(bp[1]);
      float r2 = __bfloat162float(bp[2]);
      float r3 = __bfloat162float(bp[3]);
      float lm = fmaxf(fmaxf(fabsf(r0),fabsf(r1)), fmaxf(fabsf(r2),fabsf(r3)));
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        lm = fmaxf(lm, __shfl_xor_sync(0xFFFFFFFFu, lm, off));
      if (lm < __FLT_MIN__) lm = 1.f;
      const float inv = 448.0f / lm;
      uint32_t packed = 0;
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float rv = (j==0)?r0:(j==1)?r1:(j==2)?r2:r3;
        __nv_fp8_e4m3 qv = (__nv_fp8_e4m3)(rv * inv);
        packed |= (uint32_t)(*reinterpret_cast<uint8_t*>(&qv)) << (8*j);
      }
      *reinterpret_cast<uint32_t*>(act_fp8 + (size_t)tok * K_DIM
                                   + (size_t)ksb * 128 + c) = packed;
      if (qlane == 0) act_scales[(size_t)tok * N_KBLK + ksb] = lm / 448.0f;
    }
  }

  sync_grid();  // #1  (routing + Phase-0.5 quant + fused histogram all visible)

  // ------------------------------------------------------------------
  // Phase 1b: prefix sum -> offsets, seed cursors, build tile schedule.
  // Serial on block 0 / thread 0: 256 experts + <=768 tiles is trivial
  // work; correctness-first.
  // ------------------------------------------------------------------
  // Parallel prefix-sum + tile-schedule on block 0 (one thread per expert),
  // ZERO shared memory / ZERO added barriers.  Thread e sums all lower-index
  // counts directly from GMEM (counts[] is 1KB, L2-resident) to get its own
  // exclusive row offset and tile base -- no cross-thread dependency, so no
  // scan barriers and no static SHM (which steals L1 from the compute phase).
  // Byte-identical output to the old serial loop.
  static_assert(BLOCK_SIZE == NUM_EXPERTS,
                "Phase 1b parallel assumes one thread per expert");
  if (blockIdx.x == 0) {
    // Parallel Hillis-Steele inclusive scan in dynamic SHM (log2(256)=8 steps),
    // shifted to exclusive. Replaces the O(E^2) per-thread rescan (each of 256
    // threads summing all lower counts from L2). Dynamic SHM is safe here:
    // Phase 2 fills act_cache/fp8_inter before any read, so scratching it in
    // Phase 1b cannot leak. Output byte-identical to the serial version.
    extern __shared__ char shm1b[];
    int* sc = (int*)shm1b;                       // [256] counts scan
    int* st = (int*)(shm1b + NUM_EXPERTS*4);     // [256] tilecounts scan
    const int e = static_cast<int>(threadIdx.x);
    const int cnt = ws->expert_counts[e];
    const int tiles_e = (cnt + block_m - 1) / block_m;
    sc[e] = cnt; st[e] = tiles_e;
    __syncthreads();
    #pragma unroll
    for (int d = 1; d < NUM_EXPERTS; d <<= 1) {
      int vc = (e >= d) ? sc[e - d] : 0;
      int vt = (e >= d) ? st[e - d] : 0;
      __syncthreads();
      sc[e] += vc; st[e] += vt;
      __syncthreads();
    }
    const int off_e = sc[e] - cnt;      // exclusive prefix of counts
    const int tbase = st[e] - tiles_e;  // exclusive prefix of tilecounts
    ws->expert_offsets[e] = off_e;
    ws->write_ptrs[e] = off_e;
    if (e == NUM_EXPERTS - 1) {
      ws->expert_offsets[NUM_EXPERTS] = off_e + cnt;
      ws->total_tiles = tbase + tiles_e;
    }
    for (int i = 0; i < tiles_e; ++i) {
      ws->tile_expert[tbase + i] = e;
      ws->tile_row_start[tbase + i] = off_e + i * block_m;
    }
    // Every writer publishes device-wide; grid_barrier (#3) only fences t0.
    __threadfence();
  }

  sync_grid();  // #3

  // ------------------------------------------------------------------
  // Phase 1c: scatter flattened topk indices into per-expert regions
  // ------------------------------------------------------------------
  for (int i = gtid; i < numel; i += gstride) {
    const int e = topk_ids[i];
    const int pos = atomicAdd(&ws->write_ptrs[e], 1);
    sorted_token_ids[pos] = i;
  }

  sync_grid();  // #4

  // ------------------------------------------------------------------
  // Phase 2: fused M-tiled N=64 up->SiLU->down (single-launch, intermediate in SHM)
  {
    extern __shared__ char shm[];
    uint8_t* fp8_inter = (uint8_t*)shm;
    float* inter_scales = (float*)(shm + INTER_SHM);
    // Per-row (flat id, routing weight), resolved once per tile in up_phase and
    // read again by down_phase's writeback. PERSIST-resident because it must
    // survive the phase boundary, unlike the transient union that follows it.
    int*   row_fi = (int*)  (shm + INTER_SHM + SCALE_SHM_B);
    float* row_rw = (float*)(shm + INTER_SHM + SCALE_SHM_B + A_PAD*4);
    char* trans = shm + PERSIST;
    const int num_tiles = ws->total_tiles;
    for (int tile = static_cast<int>(blockIdx.x); tile < num_tiles; tile += GRID) {
      const int e = ws->tile_expert[tile];
      const int row0 = ws->tile_row_start[tile];
      const int rows = min(BLOCK_M, ws->expert_offsets[e + 1] - row0);
      up_phase(fp8_inter, inter_scales, row_fi, row_rw, trans,
               act_fp8, act_scales, up_w_scales,
               sorted_token_ids, topk_weights, e, row0, rows, up_w_tma_desc);
      __syncthreads();
      down_phase(fp8_inter, inter_scales, row_fi, row_rw, trans,
                 down_w_fp8, down_w_scales,
                 output_accum, topk_weights, sorted_token_ids, e, row0, rows, down_w_tma_desc,
                 num_tokens * TOP_K);
      __syncthreads();
    }
  }

  sync_grid();  // #5

  // ------------------------------------------------------------------
  // Phase 3: fp32 -> bf16 cast of the accumulated output
  // ------------------------------------------------------------------
  {
    // Reduction phase: sum TOP_K per-slot partials -> bf16 output. sorted_token_ids maps slot->fi;
    // but here partials are indexed by fi=tok*TOP_K+kp, so token tok owns slots [tok*TOP_K, +TOP_K).
    // Phase 3 over FEATURE-MAJOR bf16 oa [NPASS][BS*TOP_K][N_FEAT]:
    // h -> (plane = h/N_FEAT, col = h%N_FEAT). Reads 16 contiguous bf16, which
    // stays inside one plane because N_FEAT=64 is a multiple of 16.
    constexpr int W = 16;
    constexpr int NPASS = H_DIM / N_FEAT;
    const size_t out_elems = static_cast<size_t>(num_tokens) * H_DIM;
    const __nv_bfloat16* oa16 = reinterpret_cast<const __nv_bfloat16*>(output_accum);
    const size_t nslots = (size_t)num_tokens * TOP_K;
    constexpr int HV = H_DIM / W;
    const size_t out_vec = out_elems / W;
    // Communication blocks exist only when the caller supplied a multicast alias
    // and asked for some. With tp_ncomm == 0 every block takes the compute path
    // with one chunk and a grid stride of GRID_SIZE*BLOCK_SIZE, which is the
    // shipped loop unchanged.
    const bool tp = (tp_mc != nullptr) && (tp_pads != nullptr) && (tp_ncomm > 0);
    const int ncomp = tp ? (static_cast<int>(GRID_SIZE) - tp_ncomm)
                         : static_cast<int>(GRID_SIZE);
    const int nchunk = tp ? tp_nchunk : 1;
    // sync_grid() #1 through #5 all sit between the publishing store and this
    // load, and the grid barrier carries the happens-before, so a relaxed load
    // is sufficient and no block spins on the sequence.
    const uint32_t tp_seq_eff =
        (tp_seqctr != nullptr) ? tp_ld_relaxed(&ws->tp_seq_pub) : tp_seq;

    if (!tp || static_cast<int>(blockIdx.x) < ncomp) {
      const size_t cgstride = static_cast<size_t>(ncomp) * BLOCK_SIZE;
      for (int c = 0; c < nchunk; ++c) {
        const size_t c_lo = out_vec * c / nchunk;
        const size_t c_hi = out_vec * (c + 1) / nchunk;
        for (size_t iv = c_lo + static_cast<size_t>(gtid); iv < c_hi; iv += cgstride) {
      int tok = iv / HV, hv = iv % HV;
      int h = hv * W;
      int plane = h / N_FEAT, col = h % N_FEAT;
      float s[W];
      #pragma unroll
      for (int q = 0; q < W; ++q) s[q] = 0.f;
      #pragma unroll 8
      for (int kp = 0; kp < TOP_K; ++kp) {
        // bf16 read: same 2x uint4 (32 B) loads, same coalescing; only the
        // decode differs. Accumulation stays fp32 and the final round to
        // bf16_output is unchanged, exactly as in shipped.
        const __nv_bfloat16* src = &oa16[(size_t)plane*nslots*N_FEAT +
                                        ((size_t)tok*TOP_K + kp)*N_FEAT + col];
        uint4 raw[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u) raw[u] = *reinterpret_cast<const uint4*>(src + u*8);
        const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(raw);
        #pragma unroll
        for (int q = 0; q < W; ++q) s[q] += __bfloat162float(p[q]);
      }
      __nv_bfloat16 o[W];
      // Nothing to undo: the partials were stored unscaled. The fp32 sum rounds
      // to bf16 exactly as shipped does.
      // Every pre-existing entry point forwards tp_residual == nullptr and
      // out_scale == 1.0f, and s[q] * 1.0f is exact in IEEE fp32, so that path
      // is the shipped store bit for bit.
      if (tp_residual != nullptr) {
        uint4 rraw[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u)
          rraw[u] = *reinterpret_cast<const uint4*>(&tp_residual[iv*W + u*8]);
        const __nv_bfloat16* rp = reinterpret_cast<const __nv_bfloat16*>(rraw);
        #pragma unroll
        for (int q = 0; q < W; ++q)
          o[q] = __float2bfloat16(s[q] * out_scale + __bfloat162float(rp[q]));
      } else {
        #pragma unroll
        for (int q = 0; q < W; ++q) o[q] = __float2bfloat16(s[q] * out_scale);
      }
      #pragma unroll
      for (int u = 0; u < 2; ++u)
        *reinterpret_cast<uint4*>(&bf16_output[iv*W + u*8]) =
            *reinterpret_cast<uint4*>(o + u*8);
        }
        if (!tp) continue;
        // System scope, not device scope: a communication block on a peer device
        // reads these bytes through the multicast alias.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&ws->tp_ready[c], 1u);
      }
    } else {
      const int cb = static_cast<int>(blockIdx.x) - ncomp;
      const int nsh = tp_world * tp_ncomm;
      const int sh = tp_rank * tp_ncomm + cb;
      const size_t out_mv = out_elems / TP_MM;
      for (int c = 0; c < nchunk; ++c) {
        if (threadIdx.x == 0)
          while (tp_ld_relaxed(&ws->tp_ready[c]) < static_cast<uint32_t>(ncomp))
            __nanosleep(64);
        __syncthreads();
        // Every rank has now written chunk c locally. This barrier is the
        // cross-rank release a collective would otherwise enforce for us.
        if (threadIdx.x == 0)
          tp_rank_barrier(tp_pads, tp_rank, tp_world, c * tp_ncomm + cb,
                          tp_seq_eff);
        __syncthreads();
        const size_t lo = out_mv * c / nchunk, hi = out_mv * (c + 1) / nchunk;
        // Partition on TP_ALIGN-unit groups so every shard starts on a 512-B
        // boundary and a warp's 32 x 16 B never straddles two segments. MEASURED
        // on the isolated primitive: a raw-unit partition split the time into two
        // clusters 35 us apart at 8192 tokens, and the fast cluster was exactly
        // the ncomm where the unit count happened to divide evenly by world*ncomm.
        const size_t ng = (hi - lo + TP_ALIGN - 1) / TP_ALIGN;
        const size_t s_lo = lo + (ng * sh / nsh) * TP_ALIGN;
        const size_t s_hi = lo + (ng * (sh + 1) / nsh) * TP_ALIGN;
        tp_mm_reduce(tp_mc, s_lo < hi ? s_lo : hi, s_hi < hi ? s_hi : hi);
      }
      // The local buffer is complete only once every rank's every communication
      // block has stored its shard, so this last barrier is global rather than
      // one per shard.
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        if (atomicAdd(&ws->tp_done_local, 1u) == static_cast<uint32_t>(tp_ncomm) - 1u) {
          tp_rank_barrier(tp_pads, tp_rank, tp_world, nchunk * tp_ncomm,
                          tp_seq_eff);
          __threadfence();
          atomicExch(&ws->tp_done_flag, 1u);
        } else {
          while (tp_ld_relaxed(&ws->tp_done_flag) == 0u) __nanosleep(64);
        }
      }
      __syncthreads();
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------
// Host API (ctypes-friendly)
// ---------------------------------------------------------------------
extern "C" {

size_t prefill_wgmma_workspace_bytes() { return sizeof(Workspace); }
int prefill_wgmma_num_experts() { return NUM_EXPERTS; }
int prefill_wgmma_top_k() { return TOP_K; }
int prefill_wgmma_max_tiles() { return MAX_TILES; }
int prefill_wgmma_grid_size() { return GRID; }
int prefill_wgmma_k_dim() { return K_DIM; }
int prefill_wgmma_h_dim() { return H_DIM; }
// FUSED_RENORM: 1 means the router weights were already divided by their own
// per token sum inside Phase 0, so a caller MUST NOT divide the output again.
// A binary built before this existed has no such symbol, the probe reports None,
// and that is read as "caller still owns the renorm". Dividing twice is not a
// crash, it is a quietly wrong answer, so the wrapper checks this symbol.
int prefill_wgmma_fused_renorm() { return 1; }

// Launches the full 4-phase kernel asynchronously on `stream`.
// output_accum is zeroed here (cudaMemsetAsync) before the launch; the
// workspace is likewise zeroed each launch (valid initial barrier state).
// Returns 0 on success, a negative sentinel on precondition/TMA failure,
// or a positive cudaError_t value on CUDA failure.
// Router-aware entry point. The bias may be null, in which case selection is
// unbiased. launch_prefill_moe_wgmma_q1 below keeps its exact original
// signature and forwards a null bias, so every driver already written against
// it links and behaves unchanged.
// The tensor-parallel entry point. `tp_mc` is a multicast alias over the SAME
// bytes as bf16_output on a symmetric window; `tp_pads` is the per-rank base of a
// separate symmetric window used only for the in-kernel rank barrier. That window
// must not be a handle's signal pad: MEASURED, writing the pad corrupts the state
// a reference collective keeps there and hangs it. With tp_ncomm == 0 this is the
// shipped launch, which is why the two older entry points below forward zeros.
// _tp2 is _tp plus the folded residual and the output scale. The older _tp
// below forwards nullptr and 1.0f, so its ABI and its numerics are untouched
// and a driver linked against the previous binary keeps working.
int launch_prefill_moe_wgmma_q1_tp2(
    const float* logits, int num_tokens, int block_m,
    void* workspace, size_t workspace_bytes,
    const void* act_bf16, uint8_t* act_fp8, float* act_scales,
    const uint8_t* up_w_fp8,       // [E, N_UP, K_DIM] e4m3, pair-interleaved
    const float* up_w_scales,      // [E, N_WROW, N_KBLK]
    const uint8_t* down_w_fp8,     // [E, H_DIM, N_HALF] e4m3
    const float* down_w_scales,    // [E, D_WROW, D_KBLK]
    float* output_accum,           // [num_tokens, H_DIM] fp32
    int32_t* topk_ids,             // [num_tokens, TOP_K] (output)
    float* topk_weights,           // [num_tokens, TOP_K] (output)
    int32_t* sorted_token_ids,     // [num_tokens * TOP_K] (output)
    void* bf16_output,             // [num_tokens, H_DIM] bf16 (output)
    const float* router_bias,      // [NUM_EXPERTS] fp32 or nullptr
    void* tp_mc, void* tp_pads, int tp_rank, int tp_world,
    int tp_nchunk, int tp_ncomm, uint32_t tp_seq,
    void* tp_seqctr,               // persistent 4-byte device word, or nullptr
                                   // to keep the old argument path
    const void* tp_residual,       // [num_tokens, H_DIM] bf16 partial residual
                                   // folded into the collective, or nullptr
    float out_scale,               // scale on the routed sum, 1.0f for none
    cudaStream_t stream) {
  if (workspace == nullptr || workspace_bytes < sizeof(Workspace)) return -100;
  if (num_tokens <= 0) return -101;
  // The tile schedule must match the compiled BLOCK_M, so take it from the
  // constant rather than repeating the literal. The old hardwired 48 was the
  // height the full-K activation cache allowed; the A ring sets it from A_ROWS.
  block_m = BLOCK_M;
  // Phase 2 constraints: 8-row WGMMA slices, <=8 slice accumulators.
  if (block_m < 8 || block_m > MAX_BLOCK_M || (block_m % 8) != 0) return -101;
  const long long numel = static_cast<long long>(num_tokens) * TOP_K;
  if (numel / block_m + NUM_EXPERTS > MAX_TILES) return -102;
  if (logits == nullptr || act_bf16 == nullptr || act_fp8 == nullptr || act_scales == nullptr ||
      up_w_fp8 == nullptr || up_w_scales == nullptr ||
      down_w_fp8 == nullptr || down_w_scales == nullptr ||
      output_accum == nullptr || topk_ids == nullptr ||
      topk_weights == nullptr || sorted_token_ids == nullptr ||
      bf16_output == nullptr) {
    return -104;
  }

  // Fused-epilogue argument checks. These reject rather than clamp, because a
  // clamped chunk count would silently reduce a config to one the sweep never
  // measured and still return 0.
  if (tp_mc != nullptr && tp_ncomm > 0) {
    if (tp_nchunk < 1 || tp_nchunk > TP_MAX_C) return -110;
    if (tp_ncomm >= static_cast<int>(GRID_SIZE)) return -111;
    if (tp_world < 1 || tp_rank < 0 || tp_rank >= tp_world) return -112;
    if (tp_pads == nullptr) return -113;
    // Chunk boundaries must land on whole 16-B multimem units.
    if ((static_cast<size_t>(num_tokens) * H_DIM) %
        (TP_MM * static_cast<size_t>(tp_nchunk))) return -114;
  }

  // Software grid barrier co-residency invariant: one block per SM,
  // GRID_SIZE <= SM count (see moe_grid_barrier.h safety note).
  int sm_count = 0;
  cudaError_t err =
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (static_cast<uint32_t>(sm_count) < GRID_SIZE) return -103;

  // TMA descriptors for the up and down weights (128x128 fp8 tiles,
  // SWIZZLE_128B), same encodings the spike verified.
  CUtensorMap up_tma{}, down_tma{};
  {
    uint64_t gd[2] = {(uint64_t)K_DIM, (uint64_t)E * N_UP};
    uint64_t gs[1] = {(uint64_t)K_DIM};
    uint32_t bx[2] = {128u, (uint32_t)N_FEAT}, es[2] = {1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(
        &up_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<uint8_t*>(up_w_fp8), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "up TMA fail:%d\n", (int)r);
      return -2;
    }
  }
  {
    uint64_t gd[2] = {(uint64_t)N_HALF, (uint64_t)E * H_DIM};
    uint64_t gs[1] = {(uint64_t)N_HALF};
    uint32_t bx[2] = {128u, (uint32_t)N_FEAT}, es[2] = {1u, 1u};
    CUresult r = cuTensorMapEncodeTiled(
        &down_tma, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<uint8_t*>(down_w_fp8), gd, gs, bx, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) {
      fprintf(stderr, "down TMA fail:%d\n", (int)r);
      return -3;
    }
  }

  // Dynamic SHM sized for the Phase-2 peak (spike's formula):
  //   weight(16KB) + mbar(128) + act(n_slices*1152) + silu_fp32(BM*2048)
  //   + fp8_inter(BM*512) + scales(BM*16) + down_act(1152)
  // 191,744 B at block_m=64 — needs the >48KB opt-in; still one block per
  // SM on H200 (228KB/SM), so barrier co-residency is preserved.
  // Phase 0-1 uses no dynamic SHM.
  // Fused Phase-2 uses the SHM union (persist + transient union); size the
  // launch to its total, not the old n=8 per-slice formula.
  const int shm_bytes = SHM_TOTAL;
  err = cudaFuncSetAttribute(prefill_moe_wgmma_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shm_bytes);
  if (err != cudaSuccess) return static_cast<int>(err);

  // Fresh scratchpad each launch: zeroes the barrier slots (valid initial
  // state) and everything else the kernel re-derives.
  err = cudaMemsetAsync(workspace, 0, sizeof(Workspace), stream);
  if (err != cudaSuccess) return static_cast<int>(err);

  // No output_accum memset: the privatized down-phase writeback unconditionally
  // writes every (slot, column) cell before Phase 3 reads it (each sorted slot is
  // covered by exactly one tile; the D-fragment mapping covers all 64 rows x 64
  // cols per pass x 32 passes = the full H_DIM). Zero-init was an atomicAdd-era
  // requirement that no longer applies.

  prefill_moe_wgmma_kernel<<<GRID_SIZE, BLOCK_SIZE, shm_bytes, stream>>>(
      logits, num_tokens, block_m, static_cast<Workspace*>(workspace),
      topk_ids, topk_weights, sorted_token_ids,
      static_cast<const __nv_bfloat16*>(act_bf16), act_fp8, act_scales, up_w_scales, down_w_fp8, down_w_scales,
      output_accum, static_cast<__nv_bfloat16*>(bf16_output), router_bias,
      up_tma, down_tma,
      static_cast<__nv_bfloat16*>(tp_mc), static_cast<uint32_t**>(tp_pads),
      tp_rank, tp_world, tp_nchunk, tp_ncomm, tp_seq,
      static_cast<uint32_t*>(tp_seqctr),
      static_cast<const __nv_bfloat16*>(tp_residual), out_scale);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

// The pre-existing tensor-parallel entry point, unchanged ABI, forwarding no
// residual and a unit scale. Byte for byte the previous binary's behaviour.
int launch_prefill_moe_wgmma_q1_tp(
    const float* logits, int num_tokens, int block_m,
    void* workspace, size_t workspace_bytes,
    const void* act_bf16, uint8_t* act_fp8, float* act_scales,
    const uint8_t* up_w_fp8, const float* up_w_scales,
    const uint8_t* down_w_fp8, const float* down_w_scales,
    float* output_accum, int32_t* topk_ids, float* topk_weights,
    int32_t* sorted_token_ids, void* bf16_output, const float* router_bias,
    void* tp_mc, void* tp_pads, int tp_rank, int tp_world,
    int tp_nchunk, int tp_ncomm, uint32_t tp_seq, void* tp_seqctr,
    cudaStream_t stream) {
  return launch_prefill_moe_wgmma_q1_tp2(
      logits, num_tokens, block_m, workspace, workspace_bytes, act_bf16,
      act_fp8, act_scales, up_w_fp8, up_w_scales, down_w_fp8, down_w_scales,
      output_accum, topk_ids, topk_weights, sorted_token_ids, bf16_output,
      router_bias, tp_mc, tp_pads, tp_rank, tp_world, tp_nchunk, tp_ncomm,
      tp_seq, tp_seqctr, nullptr, 1.0f, stream);
}

// The pre-existing router entry point, unchanged ABI, forwarding a null multicast
// alias and zero communication blocks. That combination takes the shipped Phase-3
// loop, so this arm remains checkable against the shipping numbers.
int launch_prefill_moe_wgmma_q1_router(
    const float* logits, int num_tokens, int block_m,
    void* workspace, size_t workspace_bytes,
    const void* act_bf16, uint8_t* act_fp8, float* act_scales,
    const uint8_t* up_w_fp8, const float* up_w_scales,
    const uint8_t* down_w_fp8, const float* down_w_scales,
    float* output_accum, int32_t* topk_ids, float* topk_weights,
    int32_t* sorted_token_ids, void* bf16_output, const float* router_bias,
    cudaStream_t stream) {
  return launch_prefill_moe_wgmma_q1_tp(
      logits, num_tokens, block_m, workspace, workspace_bytes, act_bf16,
      act_fp8, act_scales, up_w_fp8, up_w_scales, down_w_fp8, down_w_scales,
      output_accum, topk_ids, topk_weights, sorted_token_ids, bf16_output,
      router_bias, nullptr, nullptr, 0, 1, 1, 0, 0u, nullptr, stream);
}

// The original signature, forwarding a null bias. Under ROUTER_MODE 0 this is
// the shipped routing exactly.
int launch_prefill_moe_wgmma_q1(
    const float* logits, int num_tokens, int block_m,
    void* workspace, size_t workspace_bytes,
    const void* act_bf16, uint8_t* act_fp8, float* act_scales,
    const uint8_t* up_w_fp8, const float* up_w_scales,
    const uint8_t* down_w_fp8, const float* down_w_scales,
    float* output_accum, int32_t* topk_ids, float* topk_weights,
    int32_t* sorted_token_ids, void* bf16_output, cudaStream_t stream) {
  return launch_prefill_moe_wgmma_q1_router(
      logits, num_tokens, block_m, workspace, workspace_bytes, act_bf16,
      act_fp8, act_scales, up_w_fp8, up_w_scales, down_w_fp8, down_w_scales,
      output_accum, topk_ids, topk_weights, sorted_token_ids, bf16_output,
      nullptr, stream);
}

int prefill_wgmma_router_mode() { return ROUTER_MODE; }
int prefill_wgmma_n_group() { return N_GROUP; }
int prefill_wgmma_topk_group() { return TOPK_GROUP; }

// The two expert-width constants the host has to agree with. Without these the
// caller can only assume the geometry a binary was built for, and a mismatched
// assumption launches a kernel that reads the wrong bytes and still returns 0.
int prefill_wgmma_n_up() { return N_UP; }
int prefill_wgmma_n_half() { return N_HALF; }
// The tile height the guard below actually divides by. The launcher overrides
// its own block_m argument with this value, so a caller that inverts the guard
// to derive a token cap has to read it rather than assume the 64 that fused
// phase 2 uses in other builds.
int prefill_wgmma_block_m() { return BLOCK_M; }

// The three pipeline depths and the total, read back from the binary. On
// DeepSeek-V3 the down phase sets the shared memory budget, so UP_PIPE 2 and 4
// compile to the SAME SHM_TOTAL and the launcher's shared memory immediate
// cannot tell the two apart. A driver that asserts a depth it has no way to read
// is asserting its own filename, so the depths are exported here instead.
// The fused-epilogue geometry the host has to agree with. A driver that assumes
// these instead of reading them is asserting its own filename: the pad window it
// allocates would be sized for a chunk cap this binary does not have.
int prefill_wgmma_tp_max_c() { return TP_MAX_C; }
int prefill_wgmma_tp_align() { return (int)TP_ALIGN; }
int prefill_wgmma_tp_mm() { return (int)TP_MM; }
// 1 means launch_prefill_moe_wgmma_q1_tp takes a tp_seqctr pointer before the
// stream and will derive the sequence on device when it is non-null. A binary
// built before this exists has no such symbol, the probe reports None, and the
// caller must keep supplying a monotone tp_seq itself.
int prefill_wgmma_tp_has_devseq() { return 1; }
// Bytes the caller must allocate for tp_seqctr, zeroed once and then never
// touched again for the lifetime of the process.
int prefill_wgmma_tp_seq_bytes() { return (int)sizeof(uint32_t); }
// 1 means launch_prefill_moe_wgmma_q1_tp2 exists: _tp plus a bf16 residual
// pointer and an fp32 output scale, both consumed in the Phase-3 store so the
// single fused collective carries out_scale*routed + residual. A loader that
// does not find this symbol must not fold the shared expert.
int prefill_wgmma_tp_has_residual() { return 1; }
// Barrier words this configuration touches: one slot per chunk per communication
// block plus one final slot, each `world` wide.
int prefill_wgmma_tp_pad_words(int nchunk, int ncomm, int world) {
  return (nchunk * ncomm + 1) * world;
}

int prefill_wgmma_up_pipe() { return UP_PIPE; }
int prefill_wgmma_dn_pipe() { return DN_PIPE; }
int prefill_wgmma_a_stages() { return A_STAGES; }
int prefill_wgmma_shm_total() { return SHM_TOTAL; }

}  // extern "C"

