# Code map

If you only want to see what was written for this project, read the 14 files under
`vllm_integration/vllm/` and the 3 `.cu` files under `kernel/`. Everything else in
this repository is a harness, a gate or an archive.

Two files carry most of the work: `_custom_ops.py` (the loader, the launcher and
the geometry check) and `moe_runner.py` (where the kernel is offered the layer).

## The kernel

| File | What it is |
|---|---|
| `kernel/qwen_tp/src/prefill_mono_final_nb.cu` | The persistent expert kernel for Qwen3.5, TP=4, including the in-kernel TP chunk release in the output phase. Read this one first. |
| `kernel/qwen_ep/src/prefill_mono_final_nb.cu` | The same kernel made ownership-aware for EP=4: global-to-local expert id mapping, rows the rank does not own are skipped, local expert widths compiled in. |
| `kernel/deepseek_tp/src/prefill_mono_final_nb.cu` | The same architecture at DeepSeek-V3 geometry, TP=8: 7168-wide hidden, sigmoid router with per-expert bias. |
| `kernel/moe_monokernel/src/moe_grid_barrier.h` | The software grid barrier the phases are separated by. This is why the grid is one block per SM. |
| `kernel/moe_monokernel/src/ptx_utils.h` | Inline PTX helpers: async copy, multimem reduce, fences. |
| `kernel/build.sh` | The exact `nvcc` invocation, one variant at a time. |

`research_archive/kernel_diffs/` holds the two EP kernels as diffs against their TP
parents, which is the fastest way to see only what EP changed.

## The vLLM integration

Load, gate and launch:

| File | What was added |
|---|---|
| `_custom_ops.py` | The whole kernel-facing surface: `_load_prefill_monokernel` (dlopen, symbol binding), `prefill_monokernel_geometry` (reads the compiled-in geometry symbols back out of the `.so`), `monokernel_validate` (refuses a binary that does not match the weights), `moe_monokernel_prefill` and its `_out` variant plus the fake meta functions so the path is traceable, and `MOE_MONOKERNEL_PREFILL_MAX_TOKENS` which is why this is a prefill path. |
| `quantization/fp8.py` | The eligibility decision, in `Fp8MoEMethod.process_weights_after_loading` around line 755: compares every geometry field the `.so` reports against the loaded weights, decides TP vs EP, and on any mismatch declines and leaves the layer on Triton. |
| `envs.py` | The environment variables. `VLLM_PREFILL_MONOKERNEL_SO` has no default on purpose. |
| `runner/moe_runner.py` | Where the layer is offered to the kernel: `_maybe_moe_monokernel` and `_apply_quant_method`. Also `_monokernel_reduce_folded` and `_monokernel_folds_shared`, which decide whether the kernel returns an already-reduced output and whether it carries the shared-expert half, because the runner must not reduce twice. |

Qwen TP, the in-kernel collective release:

| File | What was added |
|---|---|
| `fused_moe/monokernel_tp.py` | The TP fusion plan: the token-count-to-chunk-count schedule, the number of blocks reserved for the release (`ncomm`), the launch wrapper `fused_tp_launch`, and `output_is_fused` so a caller can tell whether the reduce already happened. |
| `fused_moe/symm_output_pool.py` | One persistent, pre-rendezvoused symmetric buffer per slot. Exists because `symm_mem.rendezvous` is a collective and therefore cannot run during CUDA graph capture or on a per-request path. The file's header comment records the alternatives that were rejected. |
| `runner/shared_experts.py` | `pending_output`, a peek that does not consume. The fused epilogue has to read the shared half before the routed kernel launches, while the ordinary consumer still pops it afterwards. |

DeepSeek EP, which is not the persistent kernel:

| File | What was added |
|---|---|
| `fused_moe/experts/triton_moe.py` | The two EP switches on the Triton expert path: `_ep_ragged_enabled` (the ragged expert GEMM grid, both halves or neither) and `_ep_fast_silu_mode` with its readback. |
| `fused_moe/fast_silu_quant.py` | The pad-aware fused SiLU-and-mul plus per-block FP8 quantize kernel. Skips rows the rank does not own instead of computing and discarding them. `_supported` is the shape guard. Its gate is `validation/ep_silu_gate.py`. |
| `distributed/device_communicators/all2all.py` | `DeepEPHTAll2AllManager.num_sms`, the DeepEP communication SM count, line 178. 20 is the DeepEP default, 32 is the measured shipping value. |
| `prepare_finalize/deepep_ht.py` | Dispatch and combine instrumentation, plus the top-k compaction experiment (`VLLM_EP_TOPK_COMPACT`) and the worst-case-width path. The compaction result was negative and the switch defaults off. |
| `prepare_finalize/naive_dp_ep.py` | Timing spans on the AgRs (all-gather / reduce-scatter) EP path, which is the only EP backend that runs on this box besides DeepEP. |

Measurement, not implementation:

| File | What it is |
|---|---|
| `fused_moe/_ep_timer.py` | The per-layer timing harness that produced the kernel-level tables: named spans, the DeepEP and collective probes, and the dump writer. Entirely off unless `VLLM_EP_TIME_DUMP` is set. This is the first file to delete if any of this is upstreamed. |
| `fused_moe/modular_kernel.py` | Two wrappers, `_ep_prep` and `_ep_fin`, so `prepare` and `finalize` could be timed in full without reindenting their caller. Nothing else changed. |

## Reading the diff instead

`vllm_integration/integration.patch` is the same 14 files as a diff against public
upstream vLLM `fdab2b10b`:

```
git apply --stat vllm_integration/integration.patch
```

14 files, +3717 / -62. Four of them are new files.
