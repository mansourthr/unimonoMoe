# UniMonoMoE

Single-launch FP8 expert kernels for latency-bound MoE prefill, plus the vLLM
integration that runs them. A routed MoE layer normally costs a router kernel, a
sort, a grouped up-projection GEMM, an activation and quantization pass, a grouped
down-projection GEMM, a reduction and a collective. This kernel does the routing,
the counting sort, up projection, SiLU-and-mul, requantization and down projection
in one persistent CUDA launch that reads the block-wise FP8 checkpoint weights
directly, and on the tensor-parallel path it releases the output collective in
chunks from inside that same launch. Everything here targets **prefill**.

## What this repository contains

The three CUDA kernel variants that were measured, the vLLM changes needed to load
and gate them, the benchmark harnesses that produced the reported numbers, and the
correctness gates that had to pass before any timing was read.

The vLLM side is shipped two ways so it can be reviewed without a full tree:

- `vllm_integration/vllm/` holds the 14 final files at their upstream-relative
  paths, so each one can be read on its own.
- `vllm_integration/integration.patch` is the same content as a diff against
  public upstream vLLM commit `fdab2b10b`, so it can be applied and diffed.

## Main components

**Persistent FP8 expert kernel** (`kernel/`). One `__global__` function per
variant, launched with a grid sized to the GPU (132 blocks on H200) and structured
as phases separated by software grid barriers: route and count, prefix sum and
scatter, then the fused up / activation / down pipeline, then the output phase. The
geometry a build can run (K, H, expert widths, expert counts, router mode) is
compiled in as `constexpr` and exported as readable symbols, so the loader can
check a binary against the weights instead of assuming.

**Qwen tensor parallel.** The persistent kernel plus in-kernel TP chunk release:
the output phase hands finished column chunks to a symmetric-memory multimem
reduce while the remaining chunks are still being written. The communication
overlaps only the output tail. It does **not** overlap the main expert GEMM.

**Qwen expert parallel.** A separate build of the same kernel that is
ownership-aware: it maps global expert ids to the local slice, skips rows it does
not own, runs a ragged local schedule instead of a padded rectangle, and is
compiled against local rather than global expert memory geometry.

**DeepSeek-V3 tensor parallel.** The same persistent architecture transferred to a
different model geometry with a small model-specific surface (7168-wide hidden,
sigmoid router with expert bias, TP=8 shard widths). No architectural change.

**DeepSeek-V3 expert parallel.** A different path, and deliberately not the
persistent kernel. It keeps DeepEP all-to-all dispatch and combine and improves
three things around them: a ragged expert GEMM grid so padded rows do not occupy
grid blocks, a pad-aware fused activation and per-block quantization kernel that
skips rows the rank does not own, and a raised DeepEP communication SM count. The
DeepSeek EP results come from those three changes.

## Repository layout

```
kernel/
  build.sh                      compile one variant to <variant>/build/libprefill_mono.so
  moe_monokernel/src/           shared headers: grid barrier, PTX helpers
  qwen_tp/src/                  Qwen3.5 TP=4 build
  qwen_ep/src/                  Qwen3.5 EP=4 build, 64 local experts
  deepseek_tp/src/              DeepSeek-V3 TP=8 build
vllm_integration/
  vllm/                         the 14 changed or added files, upstream-relative paths
  integration.patch             the same diff against upstream fdab2b10b
benchmarks/
  q4_qwen.sh                    Qwen prefill/TTFT campaign, all arms
  qb_final.sh                   the frozen Qwen configuration used for the report
  c2_camp.sh                    DeepSeek combined EP+TP system campaign
  c3_kern.sh                    the same arms with per-layer kernel timing
  ep_cg_run.py                  serving boot used by the EP CUDA-graph validation
  vllm_dist_info_stub/          dist-info stub so a source tree imports as a package
validation/
  geo_readback.py               print the geometry a .so was compiled for
  tp8_fuse_ab.py                fused epilogue vs the collective it replaces, TP=8
  tp8_soak.py                   adversarial soak on the fused epilogue
  tp8_fold_gate.py              shared-expert fold correctness
  ep4_acc.py                    Qwen EP kernel vs an fp64 oracle, ranks in sequence
  ep_silu_gate.py               pad-aware activation, both grids, DeepSeek EP path
docs/
  CODE_MAP.md                   where to look, one line per file
  FLASHINFER_UPSTREAM_NOTES.md  what upstreaming the kernel would involve
research_archive/               earlier experiments and their write-ups, kept for reference
```

## Models and configurations tested

| Model | Parallel mode | World | Token shapes | Hardware |
|---|---|---|---|---|
| Qwen3.5-35B-A3B-FP8 | TP | 4 | 8192, 16384 | H200 |
| Qwen3.5-35B-A3B-FP8 | EP | 4 | 8192, 16384 | H200 |
| DeepSeek-V3 (FP8) | TP | 8 | 8192, 16384 | H200 |
| DeepSeek-V3 (FP8) | EP | 8 | 8192, 16384 | H200 |

Qwen3.5 cannot run wider than 4 ranks on this checkpoint: the gated shared
expert's `down_proj` is sharded on its input dimension, `512 / world`, and the FP8
checkpoint carries `weight_block_size [128, 128]`, so `512 / 128 = 4` is the widest
legal world. At `--tp 8` every rank fails at weight load. Absolute latencies are
therefore not comparable between the two models.

MoE-level expert parallel and MoE-level tensor parallel are mutually exclusive in
this vLLM runtime: under `enable_expert_parallel` the MoE config sets
`ep_size = dp * pcp * tp` and `tp_size = 1`. Requesting both gives the EP path.

## Key implementation notes

- **Prefill only.** Decode has not been evaluated. The loader refuses the kernel
  above `MOE_MONOKERNEL_PREFILL_MAX_TOKENS`, and no decode claim is made anywhere
  in this repository.
- **One block per SM is a requirement, not a tuning choice.** The phases are
  separated by a software grid barrier, so every block must be resident
  simultaneously. That fixes the grid at the SM count (132 on H200) and caps
  shared memory and register use per block.
- **The expert pipeline is fused in one launch.** Routing, counting sort, up
  projection, SiLU-and-mul, requantization to FP8 and down projection happen
  between grid barriers inside a single kernel, so the intermediate activation
  never round-trips to HBM as a separate tensor.
- **TP communication is released in chunks during the output tail only.** The
  output phase reserves `ncomm` blocks that issue a multimem reduce per finished
  column chunk. The main expert GEMM is not overlapped with communication.
- **The EP paths are topology-specific.** Local expert count, local widths and the
  rank's expert id range are compiled into the EP build, and the loader compares
  those against the loaded weights. A build for the wrong topology is refused
  before the kernel is entered, because a wrong rank inside a grid barrier wedges
  its peers instead of failing alone.
- **The DeepSeek EP result is not a persistent-kernel result.** It is the ragged
  expert grid, the pad-aware fused activation and the communication SM count. The
  persistent kernel is not on that path.
- **`VLLM_PREFILL_MONOKERNEL_SO` has no default.** An empty value makes the loader
  refuse and fall back to Triton. The off switch is `VLLM_USE_MOE_MONOKERNEL=0`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_USE_MOE_MONOKERNEL` | 1 | Master gate. 0 disables the kernel entirely. |
| `VLLM_PREFILL_MONOKERNEL_SO` | (empty) | Path to the compiled variant. Empty means no kernel. |
| `VLLM_PREFILL_MONOKERNEL_TP_FUSE` | 0 | Release the TP collective from inside the kernel. |
| `VLLM_PREFILL_MONOKERNEL_TP_CHUNKS` | built-in per token count | Override the `T:C` chunk schedule. |
| `VLLM_PREFILL_MONOKERNEL_TP_NCOMM` | 14 | Blocks reserved for the release. |
| `VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED` | 1 | Fold the shared-expert half into the same reduce. |
| `VLLM_EP_MASKED_SUM` | 0 | Ragged expert GEMM grid and pad-aware `moe_sum`. |
| `VLLM_EP_FAST_SILU` | 0 | Pad-aware fused activation and per-block quantize. 2 is the shipping value. |
| `VLLM_EP_NUM_SMS` | 20 | DeepEP communication SM count. 32 is the shipping value. |

Diagnostics, all off by default and none of them on a measured path:
`VLLM_EP_TIME_DUMP`, `VLLM_EP_ROUTE_DUMP`, `VLLM_EP_ROUTE_EPHYS`,
`VLLM_EP_PHASE2`, `VLLM_EP_COMB_BARRIER`, `VLLM_EP_DEEPEP_SYNC`,
`VLLM_EP_RS_CONTROL`, `VLLM_PREFILL_MONOKERNEL_TP_FOLD_DEBUG`.

Rejected experiments, kept so the measurement can be repeated rather than because
they are useful. Both are off by default and neither is on a shipped path:
`VLLM_EP_TOPK_COMPACT` (compacting the top-k slots before dispatch, which was a
regression) and `VLLM_EP_CG_WORST` (a worst-case-width dispatch for CUDA graph
capture).

## Build and run

Requires an sm_90a GPU (H200 was used), CUDA 13.0, and a vLLM source tree at
upstream commit `fdab2b10b` or a tree the patch applies to.

Build one kernel variant:

```
cd kernel
NVCC=/usr/local/cuda-13.0/bin/nvcc ./build.sh qwen_tp        # or qwen_ep, deepseek_tp
```

The result is `kernel/<variant>/build/libprefill_mono.so`. Confirm what it was
compiled for:

```
python3 validation/geo_readback.py kernel/qwen_tp/build/libprefill_mono.so
```

Apply the vLLM integration to a checkout:

```
cd <your vllm checkout>
git checkout fdab2b10b
git apply /path/to/unimonomoe/vllm_integration/integration.patch
```

Run a Qwen campaign. `QSNAP` is the model snapshot directory, `TREE` the vLLM
source tree, `EPBASE` a directory holding the DeepEP and FlashInfer site overlays
and receiving campaign output. All three are environment-specific; there are no
defaults that will work on another machine except `TREE` and `EPBASE`, which
default to `$HOME/vllm-upstream-port` and `$HOME/epbase`.

```
QSNAP=<path to Qwen3.5-35B-A3B-FP8 snapshot> \
TREE=<path to vllm source tree> \
EPBASE=<path to overlay and output root> \
T=8192 BOOTS=1 ARMS="tri fuse" ./benchmarks/q4_qwen.sh
```

The frozen Qwen configuration used for the report is `./benchmarks/qb_final.sh`,
which drives `q4_qwen.sh` with the arms and knobs fixed.

DeepSeek, system latency and then the per-layer decomposition of the same arms:

```
DSNAP=<path to DeepSeek-V3 snapshot> TREE=... EPBASE=... T=8192 ./benchmarks/c2_camp.sh
DSNAP=<path to DeepSeek-V3 snapshot> TREE=... EPBASE=... T=8192 ./benchmarks/c3_kern.sh
```

Both scripts default `KERNEL` to `benchmarks/../kernel`, so a build made by
`kernel/build.sh` is picked up without editing anything. `SHIPSO`, `QSO` and
`EPSO` override the individual binaries.

## Validation

The kernel gates are separate from the benchmarks on purpose: a timing number from
a run that failed correctness or never engaged the kernel is not a result.

Geometry, no GPU needed:

```
python3 validation/geo_readback.py kernel/deepseek_tp/build/libprefill_mono.so
```

The Qwen EP kernel against an fp64 oracle, single GPU, ranks run in sequence so a
fault is attributable to one rank:

```
QWEN_CKPT=<snapshot> EPSO=kernel/qwen_ep/build/libprefill_mono.so \
  python3 validation/ep4_acc.py
```

The shipped DeepSeek EP path, which is the pad-aware activation and the two expert
GEMM grids rather than the persistent kernel. No checkpoint and no `.so` are needed;
it builds its own weights and compares the bf16 output of the whole composition
bitwise, then poisons the skipped and the written rows in turn to show the
comparison is not vacuous:

```
TREE=<path to vllm source tree> python3 validation/ep_silu_gate.py
```

The gate for the archived DeepSeek EP persistent-kernel experiment is
`research_archive/ds3_acc.py`. It belongs to that experiment, which was a negative
result, and it does not gate the shipped DeepSeek EP path.

The fused TP epilogue, 8 ranks:

```
SO_SHIP=<unfused build> SO_FUSE=kernel/deepseek_tp/build/libprefill_mono.so \
  torchrun --nproc_per_node=8 validation/tp8_fuse_ab.py

SO_SHIP=<unfused build> SO_FUSE=<fused build> \
  torchrun --nproc_per_node=8 validation/tp8_soak.py

SO_SHIP=<unfused build> SO_FOLD=<fold build> \
  torchrun --nproc_per_node=8 validation/tp8_fold_gate.py
```

`SO_SHIP` and `SO_FOLD` are earlier compiles of the same source with the feature
under test disabled. They are required rather than defaulted, because a gate that
silently loaded the wrong side would still print a pass.

## Status

This is the final state of an internship research implementation. It was built to
answer performance questions on two specific models at two specific world sizes,
and it is honest about that: the geometry is compiled in per variant, the EP builds
are topology-specific, only prefill was evaluated, and several environment
variables exist because a measurement needed them. It will need cleanup and
generalization before any of it could be upstreamed. `docs/CODE_MAP.md` is the
short guide to the files, and `docs/FLASHINFER_UPSTREAM_NOTES.md` sketches what
moving the kernel into a library would involve.
