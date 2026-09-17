# Qwen3.5-35B-A3B-FP8: EP-aware persistent FP8 expert monokernel

Every number in this file is MEASURED on the p5 H200 box at the legal 4-rank Qwen topology. Nothing
is projected, extrapolated, or recomputed from a rounded displayed cell. Where a mechanism is
derived from source rather than isolated by measurement it is labelled DERIVED FROM SOURCE.

vLLM tree: `<gpu-host>/vllm-upstream-port`, branch `mansour/prefill-monokernel-upstream`,
HEAD `6d0332e9e6c2acdb739a989d5f3b21daec3cff3d`.

---

## Headline

**The EP-aware persistent monokernel beats stock Triton expert parallel by 7.07% at 8192 tokens and
9.70% at 16384 tokens of end-to-end prefill TTFT, 3 of 3 boots faster at both workloads, under the
shipping CUDA Graph configuration.** At the kernel level it takes **24.43% at 8192 and 22.99% at
16384** off the microseconds per routed MoE layer. It is correct against an fp64 oracle on real
weights, it engages on all four ranks, and it is CUDA-graph capture-safe.

**The baseline is stock Triton under the SAME EP topology.** `etri` runs expert parallel with the
monokernel off and NO `VLLM_EP_*` patch of any kind. It is not the EP-off tensor-parallel arm, which
would be a different parallel topology and would confound the comparison, and it is not an
already-optimized Triton arm.

| Comparison | 8K system TTFT | 16K system TTFT | 8K kernel us/layer | 16K kernel us/layer |
|---|---|---|---|---|
| stock Triton EP to EP monokernel | **+7.07%** | **+9.70%** | **+24.43%** | **+22.99%** |

One secondary fact, reported for completeness and not as the headline. The already-published EP
optimization `em1f`, which is the ragged expert grid plus pad-aware SiLU applied to the Triton path,
lands 5.11% at 8192 and 2.94% at 16384 AHEAD of the monokernel under graph capture. `em1f` is kept
in this file only as an engineering control: it says how much of the available EP win a targeted
patch to the Triton path already captures, which is most of it. Section 7 measures why. That
comparison does not change the headline, because both arms are measured against the same stock
Triton EP baseline and both beat it.

---

## 1. What code changed

Four files. Nothing in the working TP implementation was regressed or rewritten: the TP build is a
separate `.so` selected by a separate environment variable, and the five audited TP and EP arms of
the existing campaign are bit-identical to before.

| File | Delta | What it does |
|---|---|---|
| `prefill_final/prefill_nb_qw_ep4/src/prefill_mono_final_nb.cu` | 170 changed lines, 18 hunks | The EP-aware kernel. A copy of the TP source, so the TP build is untouched. |
| `vllm/_custom_ops.py` | 100 changed lines, 16 hunks | Threads `expert_map` through the op, adds the EP entry-point selection and the geometry validation. |
| `vllm/model_executor/layers/quantization/fp8.py` | 83 changed lines, 7 hunks | The weight-load eligibility gate: proves the map is the global-to-local representation, exports the EP geometry to the log line, passes the map to the kernel. |
| `/tmp/q4_qwen.sh` | 1 new arm branch, 1 new variable, 3 new MARK lines, 1 header block | Adds the `emk` arm. Additive only; the five existing arms are unchanged. |

Pre-EP backups exist on disk for both Python files: `_custom_ops.py.pre_ep4` (148,885 B) and
`fp8.py.pre_ep4` (48,917 B). A whole-tree checkpoint patch was taken before any edit:
`<gpu-host>/epmono_ckpt/tree_HEAD_20260916T130307Z.patch`, 99,088 B,
md5 `4ca559fd480f199665c396955aad05f2`.

### The expert_map contract, traced from source rather than assumed

`vllm/model_executor/layers/fused_moe/expert_map_manager.py`, `determine_expert_map` at line 22:
an `int32` tensor of shape `(global_num_experts,)` holding the local index where the expert is
resident on this rank and `-1` where it is not.

```python
expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
    0, local_num_experts, dtype=torch.int32)
```

`routed_experts.py:227-237` returns a DIFFERENT object, a 0/1 mask, when
`moe_kernel.fused_experts.consumes_expert_mask` is true, which is the AITER/ROCm path. On CUDA it
is false, so `layer.expert_map` is the `-1` map. The gate does not rely on that: it checks
`(m >= 0).sum() == local_experts` and `m.max() == local_experts - 1`, which rejects the 0/1 mask.

### The five changes inside the kernel

1. **Phase 0 routing remaps global to local.** The router still scores all 256 global experts. The
   fused histogram `atomicAdd(&ws->expert_counts[bidx], 1)` now takes `bidx` from the map, and an
   unowned slot contributes nothing.
2. **The fused renorm is deliberately left global.** `wsum` accumulates all 8 global top-k weights
   including unowned ones, so every rank computes an identical `1.0f / fmaxf(wsum, 1e-30f)`. Each
   rank's output is therefore a correctly scaled PARTIAL and the four sum to the full MoE output
   with no further normalization. This is what makes the existing cross-rank all-reduce correct
   unchanged, because `MoEPrepareAndFinalizeNoDPEPModular.output_is_reduced()` is False.
3. **Phase 1c counting sort skips unowned slots** with one `if (e < 0) continue;`. This is what
   makes the schedule ragged: no unowned row is ever placed in a tile, so no padded tile exists to
   skip later.
4. **Phase 3 skips unowned slots when summing.** `output_accum` is deliberately never memset, so
   under EP an unowned slot would read uninitialized memory. Fixed with an ownership skip rather
   than a memset, because a memset of 8192 x 8 x 2048 x 2 B = 268 MB costs about 67 us per layer at
   4 TB/s, which is 6.6% of the whole routed layer.
5. **TMA descriptor extents use the LOCAL expert count.** `gd[1] = E_LOCAL * N_UP` and
   `E_LOCAL * H_DIM` rather than `E * ...`. At `E=256` those descriptors would have described a
   weight tensor four times the size of the one actually allocated.

Phase 1b's Hillis-Steele scan was NOT changed. It carries
`static_assert(BLOCK_SIZE == NUM_EXPERTS)` and the workspace arrays stay at 256 slots indexed by
LOCAL id, so slots 64 to 255 simply hold count 0, emit 0 tiles, and thread 255 still writes the
correct `expert_offsets[256]`. Phase 2's tile loop was not changed either: all four of its
expert-strided indexings derive their stride from `N_UP`, `N_KBLK`, `H_DIM` and `N_HALF`, never
from `E`.

---

## 2. Final EP monokernel geometry, read back from the compiled binary

`<gpu-host>/prefill_final/prefill_nb_qw_ep4/build/libprefill_mono.so`,
md5 `25f11d9fc7a2b868f84c3c9b785e62b2`. Source md5 `ac940239209ce2c541d3f9e34d2dc9e9`.

| Symbol | Value |
|---|---|
| `num_experts` (global) | 256 |
| `num_experts_local` | 64 |
| `ep_capable` | 1 |
| `top_k` | 8 |
| `k_dim` | 2048 |
| `h_dim` | 2048 |
| `n_up` | 1024 |
| `n_half` | 512 |
| `block_m` | 128 |
| `router_mode` | 0 |
| `fused_renorm` | 1 |
| `grid_size` | 132 |
| `max_tiles` | 32768 |
| `shm_total` | 200064 |
| `workspace_bytes` | 265332 |
| `q1_ep`, `q1_ep2` | present |

`A_ROWS` is **128**, which is the largest legal tile height at this width, not a compromise:
`SHM_TOTAL` is 200,064 B of the 232,448 B SM90 cap, leaving 32,384 B of headroom. The old
`N_HALF=512` Qwen build's `A_ROWS=16` is a pre-A-ring artifact and was correctly not reused.
`MAX_BLOCK_M` is 128 and `BLOCK_M = A_ROWS`, so the compile-time assumptions stay consistent.

ptxas, `sm_90a`, `__launch_bounds__(256, 1)`, one block per SM by design because the kernel uses a
grid-wide barrier:

```
32 bytes stack frame, 16 bytes spill stores, 20 bytes spill loads
Used 255 registers, used 16 barriers, 32 bytes cumulative stack size
```

Build was clean, rc=0, no warnings, 4.3 s.

**No geometry compromise was required by EP.** Two things got better relative to the TP build, not
worse: spills fell from 144 B to 16 B, and the tile height is the same 128. One thing got worse:
`SHM_TOTAL` rose from 149,888 B to 200,064 B, and `GATE_PASSES = N_HALF / N_FEAT` rose from 2 to 8,
because `N_HALF` is 512 under EP against 128 under TP. That is the one compile-time constant that
degraded by 4x. It is DERIVED FROM SOURCE and was not isolated by measurement; see item 7.

The TP build for comparison reports `num_experts_local ABSENT`, `ep_capable ABSENT`,
`q1_ep ABSENT`, `n_up 256`, `n_half 128`, `shm_total 149888`.

---

## 3. Correctness status

All nine gates pass on all nine cells of the standalone harness, and all six engine-level checks
pass at 4 ranks. No performance number in this document comes from a boot that failed any gate.

### Standalone, real weights and real activations against an fp64 oracle

`/tmp/ep4_acc.py`, results `/tmp/ep4_acc_full.json`, `fails: []`. Layers 0/20/39 x batch
512/2048/8192, GPU 7 idle.

| Layer | BS | EP rel-L2 | EP cos | non-EP control rel-L2 | EP / control | bf16 floor | coverage |
|---|---|---|---|---|---|---|---|
| 0 | 512 | 1.7409e-02 | 0.9998485 | 1.7413e-02 | 0.9997 | 1.6469e-03 | 1.0000 |
| 0 | 2048 | 1.7405e-02 | 0.9998486 | 1.7410e-02 | 0.9997 | 1.6469e-03 | 1.0000 |
| 0 | 8192 | 1.7403e-02 | 0.9998486 | 1.7408e-02 | 0.9997 | 1.6469e-03 | 1.0000 |
| 20 | 512 | 2.9082e-02 | 0.9995796 | 2.9093e-02 | 0.9996 | 1.6621e-03 | 1.0000 |
| 20 | 2048 | 2.9091e-02 | 0.9995794 | 2.9102e-02 | 0.9996 | 1.6619e-03 | 1.0000 |
| 20 | 8192 | 2.9091e-02 | 0.9995794 | 2.9102e-02 | 0.9996 | 1.6618e-03 | 1.0000 |
| 39 | 512 | 2.1308e-02 | 0.9997767 | 2.1320e-02 | 0.9994 | 1.6801e-03 | 1.0000 |
| 39 | 2048 | 2.1290e-02 | 0.9997769 | 2.1303e-02 | 0.9994 | 1.6798e-03 | 1.0000 |
| 39 | 8192 | 2.1284e-02 | 0.9997770 | 2.1297e-02 | 0.9994 | 1.6799e-03 | 1.0000 |

**The bf16 output-rounding floor is the wrong yardstick for this kernel** and reading the middle
column against it would have raised a false alarm. The kernel quantizes activations to fp8 before
the up GEMM, so its distance from an fp64 oracle is about 2e-2 to 3e-2, roughly 10x to 17x the
bf16 floor, in the non-EP build as well. The correct control is a full-width NON-EP build at the
identical geometry, and against that the EP kernel is 0.03% to 0.06% BETTER at every cell.
**EP adds no measurable numerical cost.**

Boolean gates, all True at every cell: all launches returned 0; local expert id never above 63
(observed range `[-1, 62]` or `[-1, 63]`); exactly one rank owns each routed slot, with
`owned_per_rank` summing to `BS * 8` at every cell, e.g. L0/8192 `[15827, 20482, 13211, 16016]` =
65536; the recovered global routing is bitwise equal to `torch.topk`; the four ranks' normalized
weights are identical; every output element is finite. Both `output_accum` and `bf16_output` were
NaN-poisoned before every launch, so a missing Phase 3 skip or an unwritten output row would have
surfaced as NaN. Neither fired.

### Pre-entry decline

A wrong build must decline BEFORE entry, because after entering the persistent fused path a bad
rank can wedge its peers and there is no safe post-entry fallback. Two new launcher sentinels:
`-120` for an EP build handed a null `expert_map`, `-121` for a non-EP build handed a map. Both the
new EP entry point and every pre-existing entry point were called with a null map and both returned
**-120**, gate result `{'null_map': -120, 'old_abi': -120, 'pass_': True}`. This works because all
five pre-existing entry points forward through the single full-bodied launcher, which now carries
`if (EP_BUILD && expert_map == nullptr) return -120;`.

### Engine level, 4 ranks

`q4_emk_t8192_val_b0.log` paired against `q4_etri_t8192_val_b0.log`.

| Required check | Result |
|---|---|
| All four ranks report the EP monokernel engaged | PASS. `Worker_TP0_EP0` through `TP3_EP3` each log `MoE monokernel fast path ENABLED (E=256, E_LOCAL=64, ep=1, N_UP=1024, N_HALF=512, K=2048, H=2048, top_k=8, router_mode=0)`. 160 lines = 40 routed layers x 4 ranks. |
| No silent fallback | PASS. Zero `MoE monokernel DECLINED` lines, zero `-120` or `-121` returns, zero illegal-memory-access. |
| No rank uses an out-of-range local expert | PASS, standalone harness above. |
| Generated token IDs match the `etri` Triton EP baseline | PASS. Both arms `ids=[494]`, `first_token=' from'`, every rep. |
| Numerical output against the appropriate reference | PASS, 9/9 cells above against a non-EP full-width control. |
| All ranks reach completion | PASS, `rc=0`. |

The kernel is also **CUDA-graph capture-safe**: `q4_emk_t8192_gval_b0.log` reports
`num_cudagraph_captured: 41` on all four ranks under `FULL_AND_PIECEWISE` with
`splitting_ops_moe: []`, and the token IDs are still `[494]`. That is what makes the system-level
comparison below legal on the same basis as the audited arms.

The correctness test did not hang and was not relaunched.

---

## 4. Proof that all four ranks actually used the new kernel

`ep_cg_run.py` exposes no monokernel counters, so the evidence is the per-rank weight-load gate
line, and the gate now states its own geometry so the line distinguishes WHICH build ran. Both
campaign readers were extended with a dedicated gate that counts DISTINCT ranks matching

```
MoE monokernel fast path ENABLED (E=256, E_LOCAL=64, ep=1, N_UP=1024, N_HALF=512,
```

and requires the count to equal the world size, plus the inverse assertion that this pattern does
not appear on any arm that is not the EP monokernel arm. The TP build cannot match it: it reports
`N_UP=256, N_HALF=128` and exports no `E_LOCAL` symbol at all. Every one of the 27 `emk` boots in
this document passed that gate at 4 of 4 ranks.

The reader edits were regression-checked against the audited campaign before being trusted. They
reproduce the published Qwen numbers exactly: system `etri` 93.10 / 173.95 ms and `em1f` 82.05 /
151.85 ms at +12.30% / +12.43%, 3 of 3 boots; kernel `etri` 1314.0 / 2203.3 and `em1f` 1061.0 /
1629.9 us per layer at +19.25%.

---

## 5. Kernel level: microseconds per routed MoE layer, eager, 3 boots per arm

Eager is required here because the instrument is CUDA events recorded inside the model forward, and
events recorded inside a captured graph carry no per-replay timing. Cells are the mean over boots
of the per-rank mean. `VLLM_EP_TIME_DUMP` on, tag `_k3`.

| Arm | 8K us/layer | vs `etri` | 16K us/layer | vs `etri` |
|---|---|---|---|---|
| `etri` stock Triton EP, no EP patches, THE BASELINE | 1346.2 | | 2203.3 | |
| `em1f` EP patch to the Triton path, engineering control | 1088.5 | +19.12% | 1631.3 | +25.96% |
| `emk` EP monokernel | **1016.8** | **+24.43%** | **1696.2** | **+22.99%** |

`em1f` to `emk`: **+6.53%** at 8192, 3 of 3 boots faster. **−3.98%** at 16384, 0 of 3 boots faster.

The 8192 sign is not a win, and the decomposition is why.

| 8K, us/layer | routed MoE path | exposed collective | routed MoE total | reduce skew |
|---|---|---|---|---|
| `etri` | 1024.3 | 321.9 | 1346.2 | 216.2 |
| `em1f` | 741.0 | 347.5 | 1088.5 | 266.3 |
| `emk` | 768.6 | 248.2 | 1016.8 | 188.2 |

| 16K, us/layer | routed MoE path | exposed collective | routed MoE total | reduce skew |
|---|---|---|---|---|
| `etri` | 1809.5 | 393.8 | 2203.3 | 132.1 |
| `em1f` | 1209.2 | 422.2 | 1631.3 | 183.2 |
| `emk` | 1267.1 | 429.2 | 1696.2 | 247.2 |

The entire 8192 kernel-level win sits in the exposed-collective column, 248.2 against 347.5, and
that column is a per-layer arrival-skew effect: `emk`'s reduce skew is 188.2 against `em1f`'s
266.3. **It does not reproduce at 16384**, where the skew ordering flips to 247.2 against 183.2 and
the collective advantage disappears entirely. A skew finding at one workload is not a finding, so
this is not credited as a win.

What DOES reproduce at both workloads is the compute path. `emk` is slower there by
**27.6 us/layer (+3.7%) at 8192** and **57.9 us/layer (+4.8%) at 16384**. That is the real signal.

Two honest caveats. `etri` at 16384 contributes n=1 rather than n=3, because two of its three boots
wrote 0 and 3 rank dumps instead of 4 and were excluded by the reader; its surviving value of
2203.3 is identical to the audited 4-boot figure. The `em1f` to `emk` comparison, which is the one
under test, has the full n=3 at both workloads. Closure and additive checks are exact to
0.000000 us on every arm. `route`, `layer`, `prep` and `fin` do not exist on `emk` at all: routing,
both expert GEMMs and the ownership-filtered weighted reduction all happen inside one call, so
there is no inner region left to bracket. That absence is a result, not a missing measurement.

---

## 6. System level: prefill TTFT in milliseconds, 3 boots per arm

Cells are the median over boots of that boot's median timed rep, rep 0 discarded as warmup.
Improvements are the mean over common boot indices of the per-boot improvement, computed inside the
pair before averaging. The ratio of the two displayed medians is given as a cross-check only and is
not the statistic. All 18 boots of the CUDA Graph campaign passed every gate.

### CUDA Graph, the shipping configuration, tag `_s3`

| Arm | 8K ms | vs `etri` | 16K ms | vs `etri` |
|---|---|---|---|---|
| `etri` stock Triton EP, no EP patches, THE BASELINE | 93.05 | | 174.55 | |
| `em1f` EP patch to the Triton path, engineering control | 82.45 | +11.59% | 151.65 | +12.26% |
| `emk` EP monokernel | **86.35** | **+7.07%** | **156.65** | **+9.70%** |

`em1f` to `emk`: **−5.11%** at 8192 and **−2.94%** at 16384, **0 of 3 boots faster at either**.
Cross-check ratios of medians: −4.73% and −3.30%.

### Eager, the diagnostic control, tag `_e3`

Run because the kernel probe and the system rows disagreed in SIGN at 8192. This arm is whole-model
TTFT on the same eager basis as the kernel probe, with the per-layer event probe OFF so the
instrument cannot pay for itself. All 18 boots passed every gate.

| Arm | 8K ms | 16K ms |
|---|---|---|
| `etri` | 125.40 | 229.50 |
| `em1f` | 119.50 | 206.05 |
| `emk` | 116.30 | 209.75 |

`em1f` to `emk` in eager: **+2.68% at 8192** by ratio of medians, 3 of 3 boots faster, and
**−1.80% at 16384**, 0 of 3 boots faster. The mean-of-paired statistic reads +7.44% at 8192, but
`em1f` boot b1 is an outlier at 141.90 ms and dominates that mean; the two clean boots give +1.57%
and +2.38%. The honest 8192 eager figure is therefore about +2%, not +7%.

---

## 7. Why the Triton EP patch stays ahead of the monokernel, from the measurements

Everything in this section is about the SECONDARY `em1f` comparison. The headline result, stock
Triton EP to the monokernel, is a win at both workloads in both modes and is not in question here.

### The dominant term is host launch overhead, and CUDA Graph removes it from the competitor too

Put the same three arms side by side in both modes and the mechanism is visible directly:

| Arm | 8K eager ms | 8K graph ms | graph benefit | 16K eager ms | 16K graph ms | graph benefit |
|---|---|---|---|---|---|---|
| `etri` | 125.40 | 93.05 | −25.80% | 229.50 | 174.55 | −23.94% |
| `em1f` | 119.50 | 82.45 | **−31.00%** | 206.05 | 151.65 | **−26.40%** |
| `emk` | 116.30 | 86.35 | −25.75% | 209.75 | 156.65 | −25.32% |

At 8192 the graph helps `em1f` by 31.00% and `emk` by only 25.75%, a differential of 5.25 points,
and the `em1f`-to-`emk` comparison swings by 4.48 points across the same boundary, from +2.68% in
eager to −4.73% under graph. At 16384 the differential shrinks to 1.08 points and the swing shrinks
with it, from −1.80% to −3.30%.

That is the whole story of the sign flip. The Triton EP path issues 5 kernel launches per routed
layer and the monokernel issues 1, so in eager the monokernel saves real host time. Graph replay
eliminates launch overhead for both arms, the saving evaporates, and what remains is device time,
where the monokernel is measurably behind. The saving also shrinks with token count on its own,
because fixed per-launch host cost amortizes against twice the device work at 16384. Both effects
push the same way.

### It is NOT the wider EP GEMM geometry

This was the expected culprit and the measurement rejects it. Same kernel, same model, same 4
ranks, same `A_ROWS=128`, from the audited campaign versus this one, at 8192:

| Monokernel build | per-expert width | routed MoE path, us/layer |
|---|---|---|
| TP, `mono` arm | `N_UP=256`, `N_HALF=128` | 818.2 |
| EP, `emk` arm | `N_UP=1024`, `N_HALF=512` | 768.6 |

The wide EP geometry is 49.6 us/layer **faster** for the monokernel, not slower. The kernel handles
the EP shape well. `GATE_PASSES` going from 2 to 8 is real in the source but did not produce a net
regression at the whole-path level, and it was not isolated by measurement.

### It is NOT the tile height, shared memory, or spills

`A_ROWS=128` is the same as the shipping TP build, so nothing was given up. `SHM_TOTAL` is 200,064
of 232,448 B with 32,384 B of headroom, so shared memory is not at the limit. Spills went DOWN,
16 B of stores against the TP build's 144 B. Occupancy is one block per SM in both builds, which is
a requirement of the grid-wide barrier rather than a pressure symptom. None of these is the limiter.

### What the competitor already took

The reason there is so little left for the monokernel to win is that the optimized Triton EP path
already captured the EP-specific structural win, by exactly the same idea the monokernel gets by
construction. Compute path at 8192: Triton EP unoptimized 1015.7, Triton EP with the ragged expert
grid and pad-aware activation 728.5. The ragged grid took **287.2 us/layer** off the Triton path,
because with 64 local experts out of 256 global and top-k 8, only about a quarter of the routed row
slots land on any given rank and the padded grid was scheduling roughly four times the necessary
tiles.

The monokernel's counting sort is ragged by construction, so it never had that deficit. But it also
gains no further advantage from removing it. Both arms are now ragged, both fuse the activation,
and both sum only owned slots. What is left is raw per-expert GEMM efficiency at this shape, and
Triton is ahead of the persistent grid-barrier design by 3.7% to 4.8%.

### Routing and scheduling overhead did not get worse

`route` on the Triton arms costs 27.2 to 29.3 us/layer at 8192 and 40.2 at 16384. On `emk` routing
is inside the single span and cannot be read separately, but the compute-path gap of 27.6 us/layer
at 8192 is the same order as the entire Triton routing span, so routing cannot be both absorbed and
the source of the gap. The gap tracks token count (27.6 to 57.9, roughly 2.1x for 2x the tokens),
which is the signature of GEMM work rather than of fixed scheduling overhead.

### What communication remains exposed

The same all-reduce as before, and nothing else. Qwen EP resolves to
`MoEPrepareAndFinalizeNoDPEPModular` with the allgather reduce-scatter backend, so there is no
dispatch and no combine collective to overlap: `use_all2all_kernels` requires
`dp_size > 1 or pcp_size > 1 or is_sequence_parallel` and all three are false at this topology.
The exposed collective is 248.2 to 429.2 us/layer depending on arm and workload, and on `emk` it is
still a separate operation. Folding it into the kernel epilogue, which is what the TP `fuse` arm
does, is the untried lever, and it is where the remaining headroom is.

---

## 8. Does this change the presentation story or the title

**No, and it adds a second EP result rather than replacing one.** Nothing needs to be retitled or
removed.

The EP claim the presentation can now make is that a single persistent FP8 expert kernel takes 7.07%
and 9.70% of end-to-end prefill TTFT, and 24.43% and 22.99% of the routed MoE layer, off stock
Triton expert parallel at the same EP topology. That stands on its own against a baseline with no EP
patch of any kind.

The previously published `etri` to `em1f` result, +12.30% and +12.43% system, +19.25% and +26.03%
kernel, is untouched and measured against the same baseline. Both approaches beat stock Triton EP,
and this work also measures which one is further ahead and why, which is the content of section 7:
the targeted patch to the Triton path is 5.11% and 2.94% ahead of the monokernel under graph
capture, so the optimized Triton EP path is not a placeholder that a fused kernel would obviously
beat. That is now measured rather than assumed.

Two things this result supports adding, both one line rather than a section:

- The TP story and the EP story are genuinely different, and this says why in one sentence. On TP
  the monokernel wins because the fused collective removes an exposed all-reduce that Triton cannot
  remove. On EP there is no dispatch or combine collective to remove, so the only thing a single
  launch buys is host overhead, and CUDA Graph already buys that.
- The honest general lesson: part of a single-launch persistent kernel's advantage over a
  multi-launch Triton path is a host-overhead advantage, so it must be measured under CUDA Graph
  rather than in eager mode. Against stock Triton EP the win survives capture, 7.15% eager to 7.07%
  graph at 8192. Against the already-patched Triton path it does not: +2% eager becomes −4.7% under
  graph. Reporting only the eager number for that second comparison would have been wrong.

The remaining lever, if EP is ever revisited, is the one the TP arm already proved: fold the
cross-rank all-reduce into the monokernel epilogue. It is 248.2 to 429.2 us/layer of still-exposed
collective, which is 24% to 25% of the routed layer, and it is the only term where the monokernel
has a structural advantage Triton cannot match. That is a larger prize than the 3.7% to 4.8% compute
deficit is a penalty. It was not attempted here.

---

## 9. Exact paths, commits and logs

**Tree.** `<gpu-host>/vllm-upstream-port`, branch `mansour/prefill-monokernel-upstream`,
HEAD `6d0332e9e6c2acdb739a989d5f3b21daec3cff3d`, `git describe` = `v0.27.2rc0-139-g6d0332e9e`.
Pre-edit checkpoint `<gpu-host>/epmono_ckpt/tree_HEAD_20260916T130307Z.patch`, 99,088 B,
md5 `4ca559fd480f199665c396955aad05f2`.

**Kernel.** Source `<gpu-host>/prefill_final/prefill_nb_qw_ep4/src/prefill_mono_final_nb.cu`,
md5 `ac940239209ce2c541d3f9e34d2dc9e9`. Binary
`<gpu-host>/prefill_final/prefill_nb_qw_ep4/build/libprefill_mono.so`,
md5 `25f11d9fc7a2b868f84c3c9b785e62b2`. Build log
`<gpu-host>/prefill_final/prefill_nb_qw_ep4/build/build.log`. Local editing copy
`<workstation>/epmono/base_qw_tp4fuse.cu`. The untouched TP build it was copied from
is `<gpu-host>/prefill_final/prefill_nb_qw_tp4fuse/`.

**Python.** `vllm/_custom_ops.py` md5 `2d2abfd027b04209b6d54a0342813007`, backup
`_custom_ops.py.pre_ep4`. `vllm/model_executor/layers/quantization/fp8.py` md5
`4c80fabf638638d173215ddffa6d503d`, backup `fp8.py.pre_ep4`. Local copies
`<workstation>/epmono/custom_ops.py` and `.../fp8.py`.

**Correctness.** Harness `/tmp/ep4_acc.py`, local `<workstation>/epmono/ep4_acc.py`.
Results `/tmp/ep4_acc_full.json`, log `/tmp/ep4_acc_full.log`. Geometry read-back `/tmp/ep4_geo.py`,
local `<workstation>/epmono/ep4_geo.py`. Non-EP controls
`<gpu-host>/prefill_final/prefill_nb_qctl/` and `.../prefill_nb_maxtiles8192_oabf16/`.

**Campaign.** Arm script `/tmp/q4_qwen.sh`, local
`<workstation>/epmono/q4_qwen.sh`. Drivers `/tmp/q4_emk_camp.sh` and
`/tmp/q4_emk_eager.sh`, local `<workstation>/epmono/`. Driver logs
`/tmp/q4_emk_camp.log`, `/tmp/q4_emk_eager.log`, `/tmp/q4_emk_val.log`, `/tmp/q4_emk_gval.log`.

**Boot logs**, all under `<gpu-host>/epbase/q4qwen/`:

| Tag | Content | Count |
|---|---|---|
| `_val` | first correctness pair, `etri` and `emk` at 8192, eager | 2 |
| `_gval` | graph capture-safety check, `emk` at 8192 | 1 |
| `_s3` | system campaign, CUDA Graph, 3 arms x 3 boots x 2 token counts | 18 |
| `_k3` | kernel campaign, eager, `TIMEDUMP=1`, plus 4 `.rank*.json` dumps per boot | 18 |
| `_e3` | eager system control, no probe | 18 |

Naming is `q4_<arm>_t<tokens><tag>_b<boot>.log`.

**Readers.** `/tmp/q5_read.py` md5 `6fac72efee7e833b340a3bab168bfc91` for the system tags,
`/tmp/q6_read.py` md5 `373aef7e20bab4fb177c301bd03f0ddf` for the kernel tag. Local copies
`<workstation>/epmono/`. Invocations:

```
Q5TAG=_s3 Q5MODE=graph Q5ARMS='etri em1f emk' python /tmp/q5_read.py 8192 16384
Q5TAG=_e3 Q5MODE=eager Q5ARMS='etri em1f emk' python /tmp/q5_read.py 8192 16384
Q6TAG=_k3            Q6ARMS='etri em1f emk' python /tmp/q6_read.py 8192 16384
```

**Box state.** GPUs 4,5,6,7 throughout. GPU 1 carried one unrelated tenant's inference server for
the whole campaign, outside the GPU set and never touched. `MARK preflight_apps=1` on every
campaign confirms no foreign 8-GPU tenant was present.
