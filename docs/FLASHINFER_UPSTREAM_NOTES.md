# Notes on upstreaming the kernel to FlashInfer

What would have to happen for the persistent FP8 expert kernel to become a
FlashInfer op. This is a scoping document, not a plan of record. Nothing here has
been submitted anywhere.

The unit that could plausibly move is the kernel plus a thin launcher. The vLLM
changes in this repository are the integration that made it measurable and are not
themselves upstreamable: they touch a quantization method, a runner, a shared-expert
module and two EP prepare/finalize paths, which is a vLLM-shaped surface. It should
not be pushed at FlashInfer as one change.

## What would move

- `kernel/*/src/prefill_mono_final_nb.cu`, one persistent `__global__` per variant.
- `kernel/moe_monokernel/src/moe_grid_barrier.h`, the software grid barrier.
- `kernel/moe_monokernel/src/ptx_utils.h`, the async-copy / multimem / fence helpers.
- The launcher logic in `vllm/_custom_ops.py`: `_moe_monokernel_prefill_impl`
  (argument checks, scratch allocation, the entry-point choice) and
  `monokernel_validate` plus `prefill_monokernel_geometry` (the compiled-geometry
  readback and the weights comparison).

The C entry points already take plain pointers and sizes, not framework objects:
activations, router logits, the two FP8 expert weight tensors with their block
scales, output, plus the TP arguments. The weight layout expected is the one vLLM
already stores (`w13` gate/up pair-interleaved as `[E, N_UP, K]`, `w2` as
`[E, H, N_HALF]`, both with `[128, 128]` block scales), so no repacking step has to
travel with the kernel.

## What is vLLM-specific and would have to be abstracted

1. **Loading.** The kernel is loaded with `ctypes.CDLL` from a path in an
   environment variable, and the entry points are bound by symbol name. A library
   would compile it as a real extension module (FlashInfer's JIT or AOT path) and
   drop the `.so`-path variable, the dlopen and the symbol binding entirely.
2. **The geometry gate.** Today the geometry is compiled in as `constexpr` and read
   back through exported accessor functions so a Python-side gate can compare it
   against the loaded weights and decline. That design exists because the shipped
   artifact is a prebuilt `.so` whose provenance cannot be trusted. Under JIT the
   geometry is a compile argument, so the gate becomes a dispatch decision, and the
   readback symbols are only needed for a prebuilt-kernel cache.
3. **Scratch memory.** The launcher allocates the workspace, the quantized
   activation buffer, the activation scales, the top-k id and weight tensors, the
   sorted-id array and the fp32 accumulator with `torch.empty` on every call. A
   library op needs a caller-provided workspace and a size query, in FlashInfer's
   usual plan/run split. This is the single largest change on the Python side.
4. **The TP collective release.** The fused path needs a symmetric-memory window, a
   multicast alias over the same bytes, a separate symmetric window for the in-kernel
   rank barrier, and a persistent device sequence counter. Those come from
   `torch.distributed._symmetric_memory` through `symm_output_pool.py`, which also
   exists to keep the buffer alive across CUDA graph captures. The abstraction
   question is whether a library op should own a symmetric buffer pool at all, or
   take the multicast pointer and the barrier window as arguments and leave the
   lifetime to the caller. The second is the smaller op.
5. **Routing conventions.** `renormalize`, the sigmoid-with-correction-bias
   selection rule, and the `expert_map` layout (int32 `[global_num_experts]`, local
   slot or -1) are vLLM's conventions. They are reasonable conventions, but they are
   part of the op's contract and should be stated rather than inherited.
6. **Output bookkeeping.** The kernel can return an already-reduced output, and it
   can fold the shared-expert half into that reduce. The runner has to know which,
   or it reduces twice. In vLLM that is `_monokernel_reduce_folded` and
   `_monokernel_folds_shared`. A library op needs this in its return contract, not
   in a runner's properties.
7. **Diagnostics.** `_ep_timer.py`, the timing spans in `modular_kernel.py` and the
   `VLLM_EP_*` probe variables are measurement scaffolding. They should not travel.

## Current supported geometry

The geometry is a compile-time constant set, not a runtime parameter. Each `.so`
runs exactly one shape:

| | Qwen TP | Qwen EP | DeepSeek-V3 TP |
|---|---|---|---|
| K (hidden in) | 2048 | 2048 | 7168 |
| H (hidden out) | 2048 | 2048 | 7168 |
| Up width | 256 | 1024 | 512 |
| Down inner width | 128 | 512 | 256 |
| Experts | 256 global | 256 global, 64 local | 256 global |
| Router | softmax | softmax | sigmoid with per-expert bias |
| World | 4 (TP) | 4 (EP) | 8 (TP) |

Shared across all three: top-k 8, block_m 128, grid 132, FP8 e4m3 weights with
`[128, 128]` block scales, fused renormalization inside the kernel.

Also fixed by the architecture rather than by a constant:

- **One block per SM.** The phases are separated by a software grid barrier, so
  every block must be resident at once. The grid is the SM count, and shared memory
  and registers per block are capped by that. On a different SM count the grid
  changes; on a GPU where the occupancy target cannot be met, the design does not
  apply.
- **sm_90a.** WGMMA, TMA and multimem. Not Blackwell-portable as written.
- **A token-count bound.** `MOE_MONOKERNEL_PREFILL_MAX_TOKENS = 30720`, and the
  tile table has a compiled `max_tiles`.

## Prefill only

This kernel targets prefill and decode has not been evaluated. It is not a
per-shape-tuned GEMM: the win comes from removing launches, HBM round-trips and a
separate collective across a large token batch. At decode token counts the routed
GEMM is memory-bound and most of the resident grid has no work, so the grid barrier
becomes a cost rather than an enabler. Any decode claim would need its own
measurement campaign, and none was run.

## Likely work before a PR

Roughly in order:

1. Make the geometry a JIT/template parameter instead of a per-variant source tree,
   or accept a small enumerated set. Three near-identical `.cu` files is the honest
   description of the current state and is not shippable.
2. Move to a plan/run split with a caller-provided workspace and a size query.
3. Decide the TP collective story: either drop the fused release from the first PR
   and upstream the single-launch expert kernel alone, or define an op that takes
   the multicast pointer and barrier window as arguments. Dropping it makes the
   first PR much smaller and still carries most of the kernel.
4. Handle a variable SM count rather than a compiled 132, and state the occupancy
   requirement in the op's own preconditions.
5. Tests that do not depend on a vLLM checkpoint: a reference MoE in fp64, generated
   weights, and the routing conventions pinned by test rather than by comment.
6. Decide whether EP ownership belongs in the same op. The EP variant differs by
   the global-to-local mapping, the skip of unowned rows and the local width
   constants, and it returns a partial sum. That is a second op signature, not a
   flag.
7. Remove the diagnostics and the environment-variable switches. What survives
   should be function arguments.

The first PR that is worth writing is item 1 plus item 2 plus item 5, for the
single-GPU non-EP kernel: one geometry-parameterized persistent FP8 expert op with a
workspace query and its own tests. The TP release and the EP variant are follow-ups.
