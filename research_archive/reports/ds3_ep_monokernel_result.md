# DeepSeek-V3-FP8: EP-aware persistent FP8 expert monokernel, measured against stock Triton EP

One result file for the DeepSeek half of the investigation, built the same way as
`qwen_ep_monokernel_result.md`: an EP-aware persistent monokernel was built and validated,
then compared against **stock Triton under the same expert-parallel topology**, at the kernel
level in eager mode and at the system level under CUDA Graph.

vLLM tree for every number in this file: `<gpu-host>/vllm-upstream-port`, branch
`mansour/prefill-monokernel-upstream`, commit `6d0332e9e`. Model
`deepseek-ai/DeepSeek-V3` FP8 e4m3 block-quantized, 8 ranks on one p5 node, EP=8, PCP=1,
`all2all_backend=allgather_reducescatter`.

## Headline

**On DeepSeek-V3 the EP-aware persistent monokernel is 5.82% SLOWER than stock Triton
expert parallel at 8192 tokens (0 of 3 boots faster) and 1.30% faster at 16384 tokens
(3 of 3 boots faster), end-to-end prefill TTFT under the shipping CUDA Graph
configuration.** At the kernel level the same pair reads **8.72% slower at 8192** and
**2.05% faster at 16384** microseconds per routed MoE layer.

**This is the opposite sign from Qwen**, where the same design won 7.07% at 8192 and 9.70%
at 16384 against the same class of baseline. The DeepSeek result is a technically justified
negative at 8192 and a marginal positive at 16384, and the mechanism is measured rather than
asserted: the monokernel's compute path is level at 8192 and 13.50% FASTER at 16384, and it
loses that in the exposed all-reduce, whose per-layer arrival spread it increases by 2.39x
and 1.63x.

**The baseline is stock Triton under the SAME EP topology.** Arm `dtri` enables expert
parallelism the ordinary way, `enable_expert_parallel=True` on the engine, and carries **no**
`VLLM_EP_*` patch of any kind and no monokernel. The all-to-all backend is whatever the engine
resolves by default, read back as `allgather_reducescatter` on every boot of every arm. It is not the EP-off tensor-parallel arm, so the parallel topology is
held fixed across every comparison in this file. The already-optimized EP Triton arm is
present as `dep` and is reported as an engineering control only, never as the baseline.

| Comparison, prefill TTFT under CUDA Graph, MEASURED | 8192 tokens | 16384 tokens |
|---|---|---|
| stock Triton EP `dtri` to EP monokernel `demk` | **-5.82%, 0 of 3 boots** | **+1.30%, 3 of 3 boots** |

Correctness and engagement were closed BEFORE any timing was read: 9 of 9 standalone
accuracy cells pass every gate against a same-geometry non-EP control, three pre-entry
decline paths return the intended sentinels, and the engine gate line reports the EP-shaped
geometry on all 8 distinct ranks with 0 declines.

Campaign scale: **54 boots, 3 phases, every boot of every arm passed every gate, 0
exclusions.**

## 1. What code changed

**Nothing on the Python side.** DeepSeek EP resolves to the same prepare/finalize class as
Qwen, so the Python work done for Qwen carried over unchanged and only a new `.so` was
needed.

Traced from the boot log rather than assumed:

```
[cg] APPLIED all2all_backend='allgather_reducescatter' use_all2all=None ep=True pcp=1 tp=8
```

`allgather_reducescatter` with `use_all2all=None` is `MoEPrepareAndFinalizeNoDPEPModular`.
That path allgathers the token block, hands **every** rank the **global** token set plus an
`expert_map` naming which of the 256 experts that rank owns, and finishes with a
reduce-scatter style all-reduce over the partial hidden outputs. There is no dispatch
collective and no combine collective on this path, which is why `VLLM_EP_NUM_SMS` is inert
at DP=1 / PCP=1 on this configuration.

Consequences for the kernel, all of which the Qwen EP build already implements:

- The kernel receives global tokens, so routing must run over all `E=256` experts on every
  rank and the top-k weight denominator must be summed over all `TOP_K=8` **global** slots so
  every rank agrees on it.
- Ownership is applied at exactly one point, `lidx = expert_map[bidx]`, after the weight has
  already been written and summed. A slot the rank does not own contributes its weight to the
  denominator and contributes no rows to the GEMM.
- The expert half is **unsharded** under EP. `N_HALF` is the full 2048 rather than the TP=8
  shard of 256, and `N_UP` is 4096 rather than 512. That single fact drives section 3.

The only new artifact is the EP-shaped build, `prefill_nb_ds3_ep8`, built from its own source
tree with `EP_BUILD=1` and `E_LOCAL=32`. The shipping TP build is a separate tree and a
separate `.so`, and section 2 shows it is byte-distinct and unchanged.

## 2. Final EP monokernel geometry, read back from the compiled binary

MEASURED by loading each `.so` and calling its exported geometry accessors
(`/tmp/geo2.py <path>`), not read from source and not read from a comment.

| Symbol | TP=8 shipping build | EP=8 new build | non-EP control build |
|---|---|---|---|
| `num_experts` | 256 | 256 | 256 |
| `num_experts_local` | symbol ABSENT | 32 | 256 |
| `ep_capable` | symbol ABSENT | 1 | 0 |
| `n_up` | 512 | 4096 | 4096 |
| `n_half` | 256 | 2048 | 2048 |
| `block_m` (`A_ROWS`) | 128 | 64 | 64 |
| `up_pipe` | 4 | 3 | 3 |
| `shm_total` bytes | 166,272 | 218,464 | 218,464 |
| `q1_ep` / `q1_ep2` entry points | ABSENT | present | present |

Shared by all three builds, MEASURED: `top_k=8`, `k_dim=7168`, `h_dim=7168`,
`router_mode=2`, `fused_renorm=1`, `a_stages=4`, `grid_size=132`, `max_tiles=32768`,
`n_group=8`, `topk_group=4`, `workspace_bytes=265332`.

ptxas on both compiled builds, MEASURED: `16 bytes stack frame, 0 bytes spill stores,
0 bytes spill loads`, `Used 168 registers, used 16 barriers`. So the EP widths did not
introduce register spills, and spills are not the explanation for anything in section 8.

**The TP build is provably not regressed.** `prefill_nb_ds3_tp8fuse3_bm128` still reports
`n_up 512, n_half 256, block_m 128, shm_total 166272, up_pipe 4`, with
`num_experts_local`, `ep_capable`, `q1_ep` and `q1_ep2` all **absent**. Two source trees, two
`.so` files with different md5 sums, independently selectable by
`VLLM_PREFILL_MONOKERNEL_SO`.

| Build | md5 of `build/libprefill_mono.so` |
|---|---|
| `prefill_nb_ds3_ep8` (EP, used for every `demk` number here) | `f9035bb87a568a4b460da6fc0bf8b9b5` |
| `prefill_nb_ds3_ctl256` (non-EP control, correctness only) | `a98b70625f2976ac5ad400bf8c69b80a` |
| `prefill_nb_ds3_tp8fuse3_bm128` (shipping TP, untouched) | `ef9a295f18fc02a3e80bf011b4e87499` |

## 3. The EP tile height is capped at 64 rows by a hard compile-time shared memory limit

The instruction was not to reuse an old tile height blindly but to compute the actual
shared-memory requirement and find the largest legal one. That was done, and the answer is a
**compile-time proof**, not an extrapolation.

Under EP the intermediate array held in shared memory between the up and down phases is
`INTER_SHM = D_KBLK * A_BLK`, and `D_KBLK` is driven by the unsharded `N_HALF`:
`N_HALF` 256 under TP gives `D_KBLK = 2`, `N_HALF` 2048 under EP gives `D_KBLK = 16`.

DERIVED from the source constants at `A_ROWS = 128`, then CONFIRMED by building it:

| Term | Value at `A_ROWS=128`, EP geometry |
|---|---|
| `A_PAD` | 128 |
| `A_BLK` | 16,384 B |
| `INTER_SHM = D_KBLK * A_BLK` | **262,144 B** |
| SM90 shared-memory cap available to the kernel | **232,448 B** |
| `PERSIST` | 271,360 B |
| `TRANS` | 114,784 B |
| `SHM_TOTAL` | 386,400 B, i.e. 153,952 B over the cap |

`INTER_SHM` **alone** exceeds the entire budget, so no reduction in pipeline depth,
no change to `UP_PIPE` / `DN_PIPE` and no ring-depth trade can buy the taller tile back.
Building it fails, MEASURED:

```
<gpu-host>/prefill_final/prefill_nb_ds3_ep8_a128/src/prefill_mono_final_nb.cu(418):
error: static assertion failed with "fused SHM exceeds cap"
  static_assert(SHM_TOTAL <= 227*1024, "fused SHM exceeds cap");
1 error detected
```

No `.so` was produced, so the illegal build cannot be loaded by accident. The same derivation
at `A_ROWS = 64` gives `SHM_TOTAL = 218,464 B`, which is exactly what the shipped EP binary
reports back, so the derivation is validated against the binary rather than trusted.

**`A_ROWS=32` is not a usable slope probe and was deliberately not measured.**
`M_SUBS = (A_ROWS + 63) / 64` pads back to `A_PAD = 64`, so `A_BLK`, `INTER_SHM` and
`SHM_TOTAL` are all IDENTICAL at 32 and 64: a 32-row build reads back `block_m=32` with
`shm_total` still 218,464 and simply wastes half of every WGMMA tile. A 32-vs-64 comparison
would have exaggerated the tile-height slope, so the build was discarded rather than timed.

**The cost of losing the taller tile is CITED, not measured here.** When `A_ROWS` 64 to 128
was taken on the DeepSeek **TP** build it was worth -4.91% at 8192 and -8.35% at 16384 kernel
time. Those figures were measured at TP geometry, they are quoted only to size what EP gives
up, and they are not a measurement at EP geometry.

## 4. Correctness status: 9 of 9 cells pass every gate

Standalone harness `ds3_acc.py`, real DeepSeek-V3 weights and real activations, 3 layers by
3 batch sizes, on an idle GPU. The EP kernel is run once per simulated rank with that rank's
`expert_map` and the owned outputs are combined, which is what the engine does.

The yardstick is a **same-geometry non-EP control** built from the identical source with one
`constexpr` changed (`E_LOCAL` 32 to 256, `EP_BUILD` 0). A bf16 output-rounding floor cannot
isolate an EP-induced change; only this control can.

MEASURED, all 9 cells:

| Layer | Batch | control relL2 | EP relL2 | EP cosine | EP / control relL2 | routing identical | weights bit-identical | fp64 disagreements |
|---|---|---|---|---|---|---|---|---|
| 3 | 512 | 2.8884e-02 | 2.8875e-02 | 0.9995831 | 0.9997x | yes | yes | 0 |
| 3 | 2048 | 2.8846e-02 | 2.8836e-02 | 0.9995842 | 0.9996x | yes | yes | 0 |
| 3 | 8192 | 2.8796e-02 | 2.8785e-02 | 0.9995857 | 0.9996x | yes | yes | 0 |
| 30 | 512 | 4.1153e-02 | 4.1150e-02 | 0.9991530 | 0.9999x | yes | yes | 0 |
| 30 | 2048 | 4.1077e-02 | 4.1075e-02 | 0.9991561 | 0.9999x | yes | yes | 0 |
| 30 | 8192 | 4.1085e-02 | 4.1082e-02 | 0.9991558 | 0.9999x | yes | yes | 0 |
| 60 | 512 | 2.5646e-02 | 2.5643e-02 | 0.9997037 | 0.9999x | yes | yes | 0 |
| 60 | 2048 | 2.9190e-02 | 2.9189e-02 | 0.9995779 | 1.0000x | yes | yes | 0 |
| 60 | 8192 | 2.6316e-02 | 2.6305e-02 | 0.9996561 | 0.9996x | yes | yes | 1 at margin 7.607e-08 |

Row coverage is 1.0000 on every cell: every routed row the reference expects is produced.

```
9 cells, 0 failing -> /tmp/ds3_acc.json
VERDICT: ALL GATES PASS
```

**Expert-parallel sharding costs zero accuracy on this kernel.** The EP-to-control relL2
ratio is between 0.9996x and 1.0000x on all nine cells, against a control that differs by
exactly one `constexpr`.

### The single fp64 disagreement is a float-epsilon tie, and the gate proves it

On layer 60 at 8192 tokens an **fp64** reference disagrees with the kernel on exactly 1 token
of 8192, at an expert-boundary margin of **7.607e-08**. An **fp32** reference, which is the
kernel's own precision including the sigmoid, disagrees on **0 of 8192**. For scale, the
median group-decision margin over all 8192 tokens is **2.317e-02** and the 0.1 percentile is
**4.356e-05**, so a real routing defect lands five orders of magnitude away from this.

An fp64 reference is therefore not a stricter version of the same test, it is a different
test. The gate now compares routing at fp32 and adds a **stronger** secondary condition: every
token where fp64 disagrees must sit under 1e-5 at one of the two decision boundaries. A genuine
routing bug still fails loudly; a tie under float epsilon does not manufacture a failure.

### Pre-entry decline, MEASURED

There is no safe fallback once a rank enters a persistent grid-barrier kernel, so an
incompatible build must be refused before entry. All three paths return the intended
sentinel:

```
G8 pre-entry decline: EP build q1_ep with a null map rc=-120 (want -120),
                      EP build pre-existing q1_router ABI rc=-120 (want -120),
                      non-EP control handed a map rc=-121 (want -121)   PASS
```

`-120` is "EP build reached without an `expert_map`", `-121` is "non-EP build handed an
`expert_map`". Both are the first statements of the entry point, before any barrier.

## 5. Proof that all eight ranks used the new kernel

MEASURED from the validation boot `d1_demk_t8192_v0_b0` and re-checked on every timed
`demk` boot. The gate line, identical on all **8 distinct** ranks:

```
(Worker_TP7_EP7) [fp8.py:918] MoE monokernel fast path ENABLED
  (E=256, E_LOCAL=32, ep=1, N_UP=4096, N_HALF=2048, K=7168, H=7168, top_k=8,
   router_mode=2, scoring=sigmoid, backend=Fp8MoeBackend.TRITON)
```

- `E_LOCAL=32, ep=1, N_UP=4096, N_HALF=2048` is the EP-shaped geometry. The TP build reports
  `N_UP=512, N_HALF=256` and exports no `E_LOCAL` at all, so this line cannot be produced by a
  TP build. The reader gates on this exact string, counted over **distinct** rank tags,
  because the gate logs more than once per rank.
- 0 declines on all 8 ranks.
- `num_cudagraph_captured: 62`, `capture_sizes_max: 8192`,
  `max_cudagraph_capture_size: 8192`, `enforce_eager: false`, **`splitting_ops_moe: []`**,
  so the MoE is inside the captured graph and not split out of it.
- Generated token ids identical to the stock Triton EP arm of the same boot index. Every
  timed boot is gated on this.

## 6. Kernel level: microseconds per routed MoE layer, eager, 3 boots per arm

Eager by construction, because the instrument is CUDA events recorded inside the model
forward and events recorded inside a captured graph carry no per-replay timing. The probe
adds host work to the timed path, so these are not system numbers and are never quoted as
such. `layers=58` was read back from the dump, not assumed: DeepSeek-V3 has
`first_k_dense_replace: 3` of 61 layers, so 58 are routed MoE.

MEASURED, us per routed MoE layer, mean over 8 ranks, mean over 3 boots. Every row ends at
its own measured total by construction:

| Arm, T=8192 | routed MoE compute path | exposed all-reduce | routed MoE total |
|---|---|---|---|
| `dtri` stock Triton EP, THE BASELINE | 3517.0 | 977.2 | 4494.3 |
| `dep` EP Triton patch, control only | 2208.4 | 1033.4 | 3241.7 |
| `demk` EP monokernel | **3505.1** | **1380.8** | **4886.0** |

| Arm, T=16384 | routed MoE compute path | exposed all-reduce | routed MoE total |
|---|---|---|---|
| `dtri` stock Triton EP, THE BASELINE | 6682.2 | 1954.7 | 8636.8 |
| `dep` EP Triton patch, control only | 4105.1 | 2073.1 | 6178.2 |
| `demk` EP monokernel | **5780.3** | **2679.8** | **8460.1** |

Closure check, worst `|path + collective - total|` across every arm and boot:
**0.000000 us**. Additive check on the Triton arms, worst
`|route + layer + reduce + other - total|`: **0.0000 us**.

Paired improvement, mean of paired per-boot improvements, positive means the second arm is
faster:

| Pair, us per routed MoE layer | 8192 | 16384 |
|---|---|---|
| `dtri` to `demk` | **-8.72%, 0 of 3** | **+2.05%, 3 of 3** |
| `dtri` to `dep` | +27.87%, 3 of 3 | +28.47%, 3 of 3 |
| `dep` to `demk` | -50.72%, 0 of 3 | -36.93%, 0 of 3 |

Per-boot spread is tight: at 8192 `demk` reads 4884.7 / 4887.0 / 4886.3 and `dtri` reads
4495.0 / 4495.0 / 4492.8, so the ranking is not a one-boot artifact. Rank spread within a
boot is +0.05% to +0.20% on every arm.

### Which spans exist is itself a result

MEASURED inner spans, us per routed layer at 8192, where a dash means the span does not
exist:

| Arm | route | layer | prep, nested | fin, nested | reduce | other | total |
|---|---|---|---|---|---|---|---|
| `dtri` | 29.8 | 3020.2 | 127.2 | 3.2 | 977.2 | 467.0 | 4494.3 |
| `dep` | 30.0 | 1710.3 | 127.2 | 3.1 | 1033.4 | 468.1 | 3241.7 |
| `demk` | - | - | - | - | 1380.8 | - | 4886.0 |

The monokernel performs routing, both expert GEMMs and the ownership-filtered weighted
reduction inside one call, so there is no inner region left to bracket. `prep` and `fin` are
nested inside `layer` and are never added to it. There is no dispatch span and no combine
span on any arm, because this EP path has no dispatch or combine collective at all.

## 7. System level: prefill TTFT in milliseconds, 3 boots per arm

### CUDA Graph, the shipping configuration, tag `_s3`

MEASURED. Per boot value is the median of timed reps with rep 0 discarded, table cell is the
median over boots.

| Arm | 8192 ms | vs stock Triton EP | 16384 ms | vs stock Triton EP |
|---|---|---|---|---|
| `dtri` stock Triton EP, THE BASELINE | 411.95 | baseline | 846.35 | baseline |
| `dep` EP Triton patch, control only | 340.45 | +17.63%, 3 of 3 | 706.00 | +16.68%, 3 of 3 |
| `demk` EP monokernel | **437.20** | **-5.82%, 0 of 3** | **836.40** | **+1.30%, 3 of 3** |

Per boot, 8192: `dtri` 411.90 / 415.55 / 411.95, `dep` 340.45 / 339.80 / 340.60,
`demk` 437.20 / 436.95 / 437.30. Per boot, 16384: `dtri` 846.35 / 845.45 / 849.25,
`dep` 706.00 / 706.30 / 704.90, `demk` 834.55 / 836.40 / 836.95.

Ratio-of-medians cross-check agrees with the paired statistic everywhere:
`dtri` to `demk` reads -6.13% and +1.18%.

### Eager, the diagnostic control, tag `_e3`

MEASURED, probe OFF so this is a system number, not the kernel probe.

| Arm | 8192 ms | vs stock Triton EP | 16384 ms | vs stock Triton EP |
|---|---|---|---|---|
| `dtri` stock Triton EP, THE BASELINE | 413.40 | baseline | 846.40 | baseline |
| `dep` EP Triton patch, control only | 341.20 | +17.51%, 3 of 3 | 705.60 | +16.63%, 3 of 3 |
| `demk` EP monokernel | 437.10 | -5.75%, 0 of 3 | 836.45 | +1.14%, 3 of 3 |

**The DeepSeek ladder does not invert under graph capture, and graph capture is worth almost
nothing on this workload.** MEASURED, same arm eager to graph:

| Arm | 8192 eager to graph | 16384 eager to graph |
|---|---|---|
| `dtri` | 413.40 to 411.95, +0.35% | 846.40 to 846.35, +0.01% |
| `dep` | 341.20 to 340.45, +0.22% | 705.60 to 706.00, -0.06% |
| `demk` | 437.10 to 437.20, -0.02% | 836.45 to 836.40, +0.01% |

This is the discriminator that mattered on Qwen and it lands differently here. On Qwen,
graph capture took stock Triton EP from 125.40 ms to 93.05 ms at 8192, a 25.8% cut, which is
why an eager ladder could not be trusted there. On DeepSeek at 8192 the same capture is worth
0.35%, so eager and graph rank the arms identically and by nearly the same margin
(-5.75% eager, -5.82% graph). The DeepSeek prefill at these shapes is compute bound, not host
launch bound.

## 8. The whole system difference is inside the routed MoE span, and the deficit is the collective

### The routed MoE span accounts for 96% to 103% of every system difference

The eager probe measures us per routed MoE layer and the eager system arm measures whole
prefill TTFT with the probe off. Summing the per-layer difference over the 58 routed layers
and comparing it against the eager system difference is a DERIVED consistency check on where
the time lives. **It is not an end-to-end figure and is not quoted as one.**

| Pair | T | per-layer delta, us, MEASURED | summed over 58 layers, ms, DERIVED | eager system delta, ms, MEASURED | coverage |
|---|---|---|---|---|---|
| `dtri` to `demk` | 8192 | +391.7 slower | 22.72 | 23.70 slower | 95.9% |
| `dtri` to `demk` | 16384 | -176.7 faster | -10.25 | -9.95 faster | 103.0% |
| `dtri` to `dep` | 8192 | -1252.6 faster | -72.65 | -72.20 faster | 100.6% |
| `dtri` to `dep` | 16384 | -2458.6 faster | -142.60 | -140.80 faster | 101.3% |

So there is no hidden third term. Whatever the monokernel does to DeepSeek prefill, it does
it inside the routed MoE span, and the kernel probe is measuring the right thing.

### The compute path is level at 8192 and 13.50% faster at 16384

MEASURED, `dtri` to `demk`, us per routed MoE layer:

| Term | 8192 | 16384 |
|---|---|---|
| routed MoE compute path | 3517.0 to 3505.1, **-11.9 us, 0.34% faster** | 6682.2 to 5780.3, **-901.9 us, 13.50% faster** |
| exposed all-reduce | 977.2 to 1380.8, **+403.6 us, 41.30% slower** | 1954.7 to 2679.8, **+725.1 us, 37.10% slower** |
| routed MoE total | +391.7 us, 8.72% slower | -176.7 us, 2.05% faster |

Read as one sentence: **the single-launch kernel does deliver a compute win on DeepSeek, but
only at the larger shape, and at both shapes it hands back more in the exposed all-reduce
than it takes in compute at 8192 and most of what it takes at 16384.**

The 8192 result also rules out one candidate explanation immediately. Tile height cannot be
the 8192 story, because at 8192 the compute path is level with stock Triton to within 0.34%.
The tile-height cap is why the 16384 compute win is 13.50% rather than larger, not why 8192
loses.

### Why the compute win appears only at 16384, DERIVED tile arithmetic

At EP=8 each rank owns 32 of 256 experts and receives the global token set, so the routed rows
landing on one rank are `T * top_k / 8`, and each local expert's rows are padded up to
`A_ROWS = 64`:

| Quantity, DERIVED | 8192 tokens | 16384 tokens |
|---|---|---|
| routed rows per rank | 8,192 | 16,384 |
| rows per local expert, uniform case | 256 | 512 |
| tiles per local expert at 64 rows | 4 | 8 |
| tiles per rank over 32 local experts | 128 | 256 |
| persistent grid blocks, MEASURED `grid_size` | 132 | 132 |
| waves | 0.97 | 1.94 |

At 8192 the entire launch is a single wave, so the persistent grid has no second wave over
which to amortize its prologue, its grid barriers and its tail, and the measured compute path
comes out level with Triton's launch-per-expert path. At 16384 there are two waves and the
measured compute path goes 13.50% ahead. The arithmetic is DERIVED and the timings are
MEASURED; the connection between them is an interpretation consistent with both, not a
separate measurement.

### The collective grows because the monokernel arrives at it with more spread

MEASURED per-layer rank skew, defined as the slowest rank minus the fastest rank on THAT
layer, averaged over layers. This is a lower bound on arrival spread, since each rank's
per-layer value is already averaged over reps.

| Arm | 8192, `layer` skew | 8192, `reduce` skew | 16384, `layer` skew | 16384, `reduce` skew |
|---|---|---|---|---|
| `dtri` | 765.3 | 764.1 | 1698.6 | 1692.8 |
| `dep` | 856.1 | 854.5 | 1891.1 | 1882.5 |
| `demk` | span does not exist | **1825.2** | span does not exist | **2759.8** |

The Triton arms give the mirror that validates the reading: `layer` skew and `reduce` skew
agree to within 0.2% on both arms at both shapes, which is what "the collective absorbs
unequal arrival" looks like when it is real. The monokernel has no `layer` span to mirror,
and its `reduce` skew is **2.39x** the baseline at 8192 and **1.63x** at 16384.

Interpretation, consistent with the measurements and with prior measured work on this same
model, but not separately proven here: a single persistent grid-barrier launch per layer
cannot complete until its worst-loaded tile chain completes, so the layer's completion time
on a rank tracks that rank's heaviest expert directly. Triton's many small per-expert launches
interleave with host-side gaps that partially absorb the same imbalance. The prior EP budget
work on DeepSeek-V3 measured that this skew is **per layer and rotates**, that per-rank means
understate it by 10x to 11x, and that 96.6% to 89.2% of it is explained by row count, so the
imbalance driving it is real and is not a property of the monokernel. What the monokernel
changes is how much of it becomes exposed wait. Proving the mechanism outright would need a
per-rank per-layer row-count dump correlated against completion time, which was not run,
because the decision it would inform is already answered by section 3.

### What was NOT the explanation

- **Not register spills or occupancy.** ptxas reports `0 bytes spill stores, 0 bytes spill
  loads`, 168 registers, on both the EP and the control build.
- **Not a silent fallback.** All 8 distinct ranks report the EP-shaped gate line with 0
  declines on every timed boot, and token ids match the stock Triton arm.
- **Not graph capture.** Eager and graph agree to within 0.1 percentage point on the
  `dtri` to `demk` comparison at both shapes.
- **Not a contaminated node.** Every boot recorded its preflight GPU state. One foreign
  process, a peer's 2.2 GiB server on GPU 1 at 0% utilization, was present throughout all
  three phases and therefore affects every arm identically; no other tenant appeared.
- **Not a one-boot artifact.** 54 boots, 3 per arm per shape per phase, 0 gate exclusions,
  and the per-boot spread within an arm is under 0.9 ms at 8192 and under 4 ms at 16384.

## 9. Why this is the opposite sign from Qwen, and what it means for the story

Side by side, same design, same statistic, same class of baseline, MEASURED:

| Model, stock Triton EP to EP monokernel | kernel, 8192 | kernel, 16384 | system graph, 8192 | system graph, 16384 |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-FP8, EP=4 | +24.43% | +22.99% | +7.07%, 3 of 3 | +9.70%, 3 of 3 |
| DeepSeek-V3, EP=8 | -8.72%, 0 of 3 | +2.05%, 3 of 3 | -5.82%, 0 of 3 | +1.30%, 3 of 3 |

Three measured differences between the two models explain the sign flip, and all three are
properties of the model geometry rather than of the implementation:

1. **DeepSeek's EP tile height is capped at 64 by shared memory, Qwen's constraint is
   different.** DeepSeek's unsharded `N_HALF = 2048` forces `D_KBLK = 16`, and
   `INTER_SHM = 262,144 B` at 128 rows exceeds the whole 232,448 B SM90 budget on its own,
   proven by a failing `static_assert` in section 3. The shipping DeepSeek TP build runs at
   128 rows and that tile height was CITED as worth 4.91% to 8.35% of kernel time on that
   build.
2. **DeepSeek prefill at these shapes is not host launch bound, Qwen is.** Graph capture is
   worth 0.35% on DeepSeek stock Triton EP at 8192 and 25.8% on Qwen. Collapsing 32 to 256
   per-expert launches into one launch is worth much less when the launches were not the
   bottleneck.
3. **DeepSeek's per-layer expert imbalance is large and rotates**, so the exposed all-reduce
   is a bigger fraction of the routed MoE span to begin with, 21.7% at 8192 on stock Triton,
   and a design that concentrates a layer's work into one barrier-synchronized launch pays
   more of that imbalance as exposed wait.

**What this does to the presentation story, stated plainly and not acted on.** The claim
"the persistent single-launch EP monokernel is a win" is TRUE on Qwen and FALSE on DeepSeek at
8192. The defensible claim across both models is narrower and more interesting: the
single-launch EP design converts per-expert launch overhead into compute, so it wins exactly
where launch overhead dominates and where the tile height is not capped by the unsharded
expert width, and DeepSeek-V3 is the measured counterexample that shows the boundary. No
presentation file has been edited.

**Not tuned further, on purpose.** The stop condition was a structural EP geometry
limitation, and section 3 is a compile-time proof rather than a tuning result, so no sweep
was launched and no further variants were built.

## 10. Exact paths, commits and logs

vLLM tree `<gpu-host>/vllm-upstream-port`, commit `6d0332e9e`, branch
`mansour/prefill-monokernel-upstream`. Tree checkpoint
`<gpu-host>/epmono_ckpt/tree_HEAD_20260916T130307Z.patch`, 99,088 B, md5
`4ca559fd480f199665c396955aad05f2`.

| Artifact | Path |
|---|---|
| EP kernel source | `<gpu-host>/prefill_final/prefill_nb_ds3_ep8/src/prefill_mono_final_nb.cu` |
| EP `.so` used for every `demk` number | `<gpu-host>/prefill_final/prefill_nb_ds3_ep8/build/libprefill_mono.so`, md5 `f9035bb87a568a4b460da6fc0bf8b9b5` |
| non-EP control `.so` (correctness only) | `<gpu-host>/prefill_final/prefill_nb_ds3_ctl256/build/libprefill_mono.so`, md5 `a98b70625f2976ac5ad400bf8c69b80a` |
| shipping TP `.so`, untouched | `<gpu-host>/prefill_final/prefill_nb_ds3_tp8fuse3_bm128/build/libprefill_mono.so`, md5 `ef9a295f18fc02a3e80bf011b4e87499` |
| illegal 128-row build, no `.so` produced | `<gpu-host>/prefill_final/prefill_nb_ds3_ep8_a128/`, build log `/tmp/ds3a128_build.txt` |
| geometry readback tool | `/tmp/geo2.py <path to .so>` |
| correctness harness | `<workstation>/epmono/ds3_acc.py`, p5 copy `<gpu-host>/epbase/ds3_acc.py` |
| correctness log and json | `<gpu-host>/epbase/ds3_acc_v2.log`, `/tmp/ds3_acc.json` |
| routing tie diagnostic | `<workstation>/epmono/ds3_g4diag.py` |
| campaign script | `<gpu-host>/epbase/d1_ds3.sh`, md5 `9b40d2aaf9b7d671937b2c0890ed2094` |
| campaign driver, 3 phases, 54 boots | `<gpu-host>/epbase/d1_camp.sh`, log `<gpu-host>/epbase/d1_camp.log` |
| boot logs | `<gpu-host>/epbase/d1ds3/d1_<arm>_t<T><tag>_b<n>.log` |
| system reader | `<workstation>/epmono/d1_read.py`, p5 copy `<gpu-host>/epbase/d1_read.py` |
| kernel reader | `<workstation>/epmono/d2_read.py`, p5 copy `<gpu-host>/epbase/d2_read.py` |

Reproduce the tables in this file:

```
cd <gpu-host>/epbase
D1TAG=_s3                /opt/pytorch/bin/python d1_read.py 8192 16384   # section 7 graph
D1TAG=_e3 D1MODE=eager   /opt/pytorch/bin/python d1_read.py 8192 16384   # section 7 eager
D2TAG=_k3                /opt/pytorch/bin/python d2_read.py 8192 16384   # section 6 kernel
ABGPU=7 /opt/pytorch/bin/python ds3_acc.py                               # section 4
```

Arm definitions, so no arm label has to be trusted:

All three arms run `EP=1 PCP=1 TP=8` with the same token count, the same `gpu_memory_utilization`
and the same capture mode inside a phase, interleaved arm by arm within each boot index so a
drift in the box hits every arm of a pair equally.

| Arm | Environment, beyond the common EP setup |
|---|---|
| `dtri` | `VLLM_USE_MOE_MONOKERNEL=0` and nothing else. No `VLLM_EP_*` variable of any kind. THE BASELINE. |
| `dep` | `VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 VLLM_EP_NUM_SMS=32`. Engineering control only. |
| `demk` | `VLLM_USE_MOE_MONOKERNEL=1 VLLM_PREFILL_MONOKERNEL_SO=<EP .so>`. No `VLLM_EP_*` patch. |

Every EP and monokernel variable is explicitly `unset` at the top of each boot, and
`VLLM_USE_MOE_MONOKERNEL=0` is exported before the arm case, because an unset
`VLLM_PREFILL_MONOKERNEL_SO` loads a stale default `.so` rather than disabling the kernel.

Phase timing: 54 boots, started 2026-09-16T17:06:52Z, finished 2026-09-16T18:32:53Z.

### Existing DeepSeek arms that were rejected as the baseline, and why

Confirmed from the scripts, not from arm names:

| Existing arm | Why it is not a stock Triton EP graph baseline |
|---|---|
| `gbase` in `ep_all6.sh` | carries `VLLM_EP_CG_WORST=1 VLLM_EP_MASKED_SUM=1`, so it is already patched, and it is a decode arm in ms per token |
| `ebase` in `ep_all6.sh` | stock, but EAGER, and a decode arm |
| `cbase` in `ep_all6.sh` | stock, but capture OFF, and a decode arm |
| `tri` in `c2_camp.sh` | EP is OFF, which changes the parallel topology |
| `ep` in `c2_camp.sh` | carries `VLLM_EP_MASKED_SUM=1 VLLM_EP_FAST_SILU=2 VLLM_EP_NUM_SMS=32`, already optimized |

None of them is a stock Triton EP arm with CUDA Graph at the prefill shapes in question, so
`dtri` was built explicitly for this comparison rather than borrowed.
