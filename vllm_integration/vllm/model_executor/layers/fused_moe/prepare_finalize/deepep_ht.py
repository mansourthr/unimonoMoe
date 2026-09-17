# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import atexit
import os
from collections.abc import Callable

import deep_ep
import torch
import triton
import triton.language as tl

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_get_previous_event,
    dbo_switch_to_comm,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

_EP_TOPK_COMPACT = int(os.environ.get("VLLM_EP_TOPK_COMPACT", "") or 0)
# Ask DeepEP for a static worst-case recv width so `intranode_dispatch` skips its
# host spin and becomes CUDA-graph capturable. 0 = in-tree dynamic width.
_EP_CG_WORST = int(os.environ.get("VLLM_EP_CG_WORST", "") or 0)
# Read back, not assumed: the ragged expert GEMM grid lives in triton_moe.py behind
# its own env gate, and without it the worst-case width costs ~2x GEMM rows. Refusing
# here is what keeps a graph baseline and a graph optimized arm on the same padding
# cost.
_EP_CG_RAGGED = int(os.environ.get("VLLM_EP_MASKED_SUM", "") or 0)
if _EP_CG_WORST and not _EP_CG_RAGGED:
    raise RuntimeError(
        "VLLM_EP_CG_WORST=1 requires VLLM_EP_MASKED_SUM=1. num_worst_tokens pads "
        "the recv buffer to local_tokens x num_ranks (2.04x real rows at T=8192), "
        "and without the ragged expert GEMM grid those padded rows still occupy "
        "grid blocks, so a graph arm without it pays ~2x expert-GEMM rows. Under "
        "CUDA graphs the ragged grid is a requirement of the only "
        "capture-compatible dispatch mode, and it must be set on the baseline arm "
        "too or the comparison is not apples-to-apples."
    )
if _EP_CG_WORST and _EP_TOPK_COMPACT:
    raise RuntimeError(
        "VLLM_EP_CG_WORST=1 is incompatible with VLLM_EP_TOPK_COMPACT=1: the "
        "compaction receiver sizes its output with a .tolist() on a device tensor, "
        "which is a host sync and is illegal inside CUDA graph capture."
    )
_EP_CG_LOGGED = [False]
_logger = init_logger(__name__)
# Reported at process exit rather than on the first call, because the first
# _receiver call in a profiling boot is a dummy pass whose width is not the width
# the engine runs. The histogram is keyed by the width the DATA produced, so a model
# whose routing does not compact shows up as width == full instead of silently
# looking like a win.
_EP_COMPACT_HIST: dict = {}
# calls, owned slots, full width slots, widest row rectangle slots
_EP_COMPACT_SUM = [0, 0, 0, 0]


def _ep_compact_report():
    """Three slot counts, so the rectangle's own waste is visible.

    full    what the tree feeds the GEMM today, recv tokens * top_k.
    rect    what this patch feeds it, recv tokens * the WIDEST row.
    owned   what a ragged layout would feed it, the exact owned slot total.

    rect vs owned is the headroom this patch does NOT recover, and it is only zero
    when every row happens to own the same number of slots.
    """
    if not _EP_COMPACT_HIST:
        return
    calls, owned, full, rect = _EP_COMPACT_SUM
    dist = " ".join(f"w{w}:{_EP_COMPACT_HIST[w]}" for w in sorted(_EP_COMPACT_HIST))
    mean_w = sum(w * c for w, c in _EP_COMPACT_HIST.items()) / max(calls, 1)
    _logger.info(
        "EP topk compaction arm %d, %d calls, width histogram %s, mean width %.2f",
        _EP_TOPK_COMPACT,
        calls,
        dist,
        mean_w,
    )
    if owned:
        _logger.info(
            "EP topk compaction slot rows: full %d, widest row rectangle %d "
            "(%.1f%% of full), exact owned %d (%.1f%% of full, %.1f%% of the "
            "rectangle), so a ragged layout would remove a further %.2fx",
            full,
            rect,
            100.0 * rect / full,
            owned,
            100.0 * owned / full,
            100.0 * owned / max(rect, 1),
            rect / max(owned, 1),
        )


atexit.register(_ep_compact_report)


@triton.jit
def _ep_amax_kernel(ids_ptr, out_ptr, n, FULL: tl.constexpr, BR: tl.constexpr):
    """Widest row and total owned slots, in one kernel and one 8 byte buffer.

    out[0] is the max over rows of the owned slot count, which is the width the
    compacted tensor needs. out[1] is the sum, which is the number of rows a fully
    ragged layout would need. Both come back in a single host copy, so recording the
    ragged total costs nothing on top of the width the pack already has to know.

    The caller must zero the buffer. An atomic_max against a stale value returns
    a width that is too WIDE, which shows up as a smaller win rather than as a bug.
    """
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, FULL)
    m = rows[:, None] < n
    ids = tl.load(ids_ptr + rows[:, None] * FULL + c[None, :], mask=m, other=-1)
    cnt = tl.sum((ids != -1).to(tl.int32), axis=1)
    tl.atomic_max(out_ptr, tl.max(cnt, axis=0))
    tl.atomic_add(out_ptr + 1, tl.sum(cnt, axis=0))


@triton.jit
def _ep_pack_kernel(
    ids_ptr,
    wts_ptr,
    oid_ptr,
    owt_ptr,
    W,
    FULL: tl.constexpr,
    FAKE,
    OFFSET,
    HAS_W: tl.constexpr,
):
    """Move each row's owned slots to the front, one program per row, one pass.

    The output row is built in REGISTERS and stored once. Two stores from different
    lanes to the same column would race: lane j writing the not owned filler and
    lane i writing a real id are not ordered within a program.

    The gather is a select over the FULL columns rather than a sort: output column j
    takes the source column whose exclusive owned count equals j. FULL is 8 here, so
    the FULL x FULL compare is 64 lanes and is cheaper than any sort. MEASURED at
    7.9 to 10.4 us on the real shapes, which is LESS than the torch.where it
    replaces.
    """
    r = tl.program_id(0)
    c = tl.arange(0, FULL)
    ids = tl.load(ids_ptr + r * FULL + c)
    v = ids != -1
    pos = tl.cumsum(v.to(tl.int32), axis=0) - 1  # rank of each owned slot
    sel = (pos[None, :] == c[:, None]) & v[None, :]
    hit = tl.sum(sel.to(tl.int32), axis=1) > 0
    oid = tl.sum(tl.where(sel, (ids + OFFSET)[None, :], 0), axis=1)
    keep = (c < W) & hit
    tail = (c < W) & (hit == 0)
    tl.store(oid_ptr + r * W + c, oid, mask=keep)
    # the tail of a short row keeps the not owned id, exactly as the default path,
    # so expert_map turns it back into -1 and the row is zeroed
    tl.store(oid_ptr + r * W + c, FAKE, mask=tail)
    if HAS_W:
        wts = tl.load(wts_ptr + r * FULL + c)
        owt = tl.sum(tl.where(sel, wts[None, :], 0.0), axis=1)
        tl.store(owt_ptr + r * W + c, owt, mask=keep)
        # 0.0 rather than the original weight: the tail id is not owned so the
        # weight is never read, and a zero cannot contribute even if it were
        tl.store(owt_ptr + r * W + c, 0.0, mask=tail)


def _default_expert_topk(ids, num_experts, offset):
    return torch.where(ids == -1, num_experts - 1 if offset == 0 else 0, ids + offset)


def _compact_expert_topk_eager(ids, weights, num_experts, offset):
    """The first version, kept selectable so its cost stays measurable.

    MEASURED at +134.0 us (4019 recv tokens), +146.8 us (8127) and +131.9 us
    (ragged, width 5) over the default torch.where, against the Triton path's
    +36.5 us on the same three shapes.
    """
    valid = ids != -1
    full = ids.size(1)
    order = torch.argsort((~valid).to(torch.int32), dim=1, stable=True)
    w = int(valid.sum(1).max().item())
    if 0 < w < full:
        ids = ids.gather(1, order)[:, :w].contiguous()
        if weights is not None:
            weights = weights.gather(1, order)[:, :w].contiguous()
        valid = ids != -1
    _EP_COMPACT_HIST[w] = _EP_COMPACT_HIST.get(w, 0) + 1
    _EP_COMPACT_SUM[0] += 1
    fake = num_experts - 1 if offset == 0 else 0
    return torch.where(valid, ids + offset, fake), weights


def _compact_expert_topk(ids, weights, num_experts, offset):
    """Narrow the topk width to the slots this rank actually owns.

    Returns (ids, weights) with the SAME global id convention the default path
    produces: owned slots offset into the global expert space, any remaining tail
    slot left at the not owned id that expert_map turns back into -1.

    Two kernels and one 8 byte host copy. The rows are partitioned, not sorted, and
    the relative order of the owned slots is preserved because the weight column is
    cut with the same permutation; a reorder that did not preserve it would pair a
    weight with the wrong expert while still producing a plausible looking tensor.
    """
    n, full = ids.shape
    fake = num_experts - 1 if offset == 0 else 0
    buf = torch.zeros(2, dtype=torch.int32, device=ids.device)
    br = 128
    _ep_amax_kernel[(triton.cdiv(n, br),)](ids, buf, n, FULL=full, BR=br, num_warps=4)
    w, owned = (int(x) for x in buf.tolist())
    _EP_COMPACT_HIST[w] = _EP_COMPACT_HIST.get(w, 0) + 1
    _EP_COMPACT_SUM[0] += 1
    _EP_COMPACT_SUM[1] += owned
    _EP_COMPACT_SUM[2] += n * full
    _EP_COMPACT_SUM[3] += n * max(w, 1)
    w = max(w, 1)
    if w >= full:
        return _default_expert_topk(ids, num_experts, offset), weights
    oid = torch.empty(n, w, dtype=ids.dtype, device=ids.device)
    has_w = weights is not None
    owt = (
        torch.empty(n, w, dtype=weights.dtype, device=weights.device) if has_w else ids
    )
    _ep_pack_kernel[(n,)](
        ids,
        weights if has_w else ids,
        oid,
        owt,
        w,
        FULL=full,
        FAKE=fake,
        OFFSET=offset,
        HAS_W=has_w,
        num_warps=1,
    )
    return oid, (owt if has_w else None)


class DeepEPHTPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP High-Throughput kernels.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        # Round up hidden size so it is compatible with DeepEP High Throughput
        # kernels.
        # DeepEP intranode kernels make copies in units of,
        # 32(warp-size) int4 elements. Round up hidden size to respect this.
        # For example, an input hidden size of 2880 with dtype torch.bfloat16
        # will be rounded up to 3072.
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer: deep_ep.Buffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.async_prepare = True
        self.sync_dbo_comm = current_platform.is_rocm()

        # The dispatch function returns a handle that the combine function
        # requires. Under DBO microbatching we must track one handle per
        # micro-batch to avoid races between threads.
        self.handles = [None, None]

        # From https://github.com/deepseek-ai/DeepEP/blob/9fe9021f29c9083cd1808ab36b740208524d9f63/deep_ep/buffer.py#L164
        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

    def _sync_dbo_comm_if_needed(self) -> None:
        if self.sync_dbo_comm and dbo_enabled():
            # ROCm DeepEP HT dispatch/combine reuse Buffer-owned communication
            # workspace. Do not let the next DBO ubatch reuse that workspace
            # before this ubatch's HT kernel has completed.
            torch.cuda.current_stream().synchronize()

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _get_dispatch_config(self) -> deep_ep.Config | None:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_dispatch_config(self.num_dispatchers_)

    def _get_combine_config(self) -> deep_ep.Config | None:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_combine_config(self.num_dispatchers_)

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        has_scales = token_scales is not None

        # Capture a DeepEP event on the compute stream before yielding.
        # This must happen before the yield so the event only covers this
        # ubatch's compute work. If captured after, the compute stream tail
        # may include the other ubatch's work, preventing overlap.
        previous_event = dbo_get_previous_event(self.buffer.capture)

        # We yield before launching the dispatch kernel since the dispatch
        # kernel will block the CPU so we want to queue up all the compute
        # for the other ubatch before the dispatch kernel starts.
        dbo_yield_and_switch_from_compute_to_comm()

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            dispatch_expert_num_tokens,
            is_token_in_rank,
            event,
        ) = self.buffer.get_dispatch_layout(
            topk_idx=rank_topk_ids,
            num_experts=num_experts,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        # Worst case is every local token landing on every rank. It is a bound, not
        # a prediction: DS3 routing sends a token to exactly 4 of 8 ranks, so ~2.04x
        # of this is real. The bound has to be static because a static shape is the
        # whole point.
        #
        # Gated on actual capture, not on the env flag alone, and the reason is
        # measured. `profile_run` calls `_dummy_run(self.max_num_tokens)` and under
        # PCP that width is not sharded, so `tokens.size(0)` is the global 8192 there
        # rather than the per-rank 1024 a real forward sees. The bound then becomes
        # 8192 x 8 = 65536 and fused_experts' workspace2 becomes
        # 65536 x 8 x 4096 x 2 = exactly 4.00 GiB, which is the allocation that OOM'd
        # the first attempt at both GPUUTIL 0.86 and 0.80. Keying on
        # `is_current_stream_capturing()` means the worst-case width appears only
        # where it is actually needed, and it also keeps the memory-profiling pass
        # byte-for-byte the same as the baseline arm, so both arms size their KV cache
        # identically.
        _nworst = 0
        if _EP_CG_WORST and torch.cuda.is_current_stream_capturing():
            _nworst = tokens.size(0) * self.num_dispatchers_
            if not _EP_CG_LOGGED[0]:
                _EP_CG_LOGGED[0] = True
                _logger.info(
                    "EP cg worst-case dispatch ACTIVE: nworst=%d local_tokens=%d "
                    "ranks=%d ragged_grid=%d",
                    _nworst,
                    tokens.size(0),
                    self.num_dispatchers_,
                    _EP_CG_RAGGED,
                )
        (
            token_data,
            expert_topk_ids,
            expert_topk_weights,
            expert_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.buffer.dispatch(
            x=token_data,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=dispatch_expert_num_tokens,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            # expert_alignment rounds the number of tokens per expert
            # to this value.
            expert_alignment=1,
            # 0 keeps the in-tree dynamic width and its host spin. > 0 takes
            # deep_ep.cpp:441's no-sync branch.
            num_worst_tokens=_nworst,
            config=self._get_dispatch_config(),
            previous_event=previous_event,
            async_finish=self.async_prepare and not dbo_enabled(),
            allocate_on_comm_stream=False,
        )

        self._sync_dbo_comm_if_needed()

        # record the handle for this ubatch
        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        dbo_switch_to_compute_sync()

        return lambda: self._receiver(
            event,
            has_scales,
            token_data,
            expert_topk_ids,
            num_experts,
            expert_num_tokens_per_expert_list,
            expert_topk_weights,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
        )

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        token_data: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        expert_topk_ids: torch.Tensor | None,
        num_experts: int,
        expert_num_tokens_per_expert_list: list[int],
        expert_topk_weights: torch.Tensor | None,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if has_scales:
            expert_x, expert_x_scale = token_data
        else:
            expert_x, expert_x_scale = token_data, None

        # The existing MOE kernels assume that all entries of topk_ids are
        # valid. To that effect, set the -1s in expert_topk_ids to some expert
        # outside this rank so the expert_map can remap it to -1 when safe.
        # With Expert Parallel, the experts are divided amongst the rank
        # sequentially. For rank 0, set it to num_experts - 1 and for all other
        # ranks set it to 0 as we know that expert_map will have a -1 in those
        # regions for those ranks.
        #
        # DeepEP's topk_ids output refers to the local experts directly. Offset
        # the topk_ids to move it back to the global experts space so it aligns
        # with existing vLLM interfaces.
        assert expert_topk_ids is not None
        if _EP_TOPK_COMPACT and expert_topk_ids.numel() > 0:
            # Same ids the else branch produces, minus the slots that would only
            # ever be zeroed. See _compact_expert_topk.
            fn = (
                _compact_expert_topk
                if _EP_TOPK_COMPACT == 1
                else _compact_expert_topk_eager
            )
            expert_topk_ids, expert_topk_weights = fn(
                expert_topk_ids,
                expert_topk_weights,
                num_experts,
                self.rank_expert_offset,
            )
        else:
            expert_topk_ids = torch.where(
                expert_topk_ids == -1,
                num_experts - 1 if self.rank_expert_offset == 0 else 0,
                expert_topk_ids + self.rank_expert_offset,
            )

        if _EP_CG_WORST and not expert_num_tokens_per_expert_list:
            # The second condition is a readback, not belt and braces. The
            # worst-case width is applied only while the stream is capturing, so on
            # a non-captured pass this same receiver runs with a populated list and
            # must take the in-tree branch. Keying on the list DeepEP actually
            # returned is what makes both paths correct from one gate.
            #
            # With num_worst_tokens > 0 DeepEP returns an empty per-expert count
            # list (buffer.py:350-352), because producing it is the CPU sync this
            # arm removes. Passing None is not a shortcut: measured by grep,
            # `expert_tokens_meta` appears in experts/triton_moe.py only at lines
            # 263, 287 and 721, all of them parameter declarations, and is never
            # dereferenced; modular_kernel.py only plumbs it through, and its own
            # docstring at line 879 says the field is "for batched" experts. The
            # contiguous HT path builds its grid from moe_align_block_size over
            # topk_ids instead. So there is nothing to recompute on GPU here.
            expert_tokens_meta = None
        else:
            # Makes a GPU-CPU copy.
            # TODO (varun): Maybe it is better to re-compute the expert_num_tokens
            # on GPU.
            expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
                expert_num_tokens_per_expert_list, device=expert_x.device
            )

        # * For non-block quant, dispatch in b16 and quantize now as
        #   DeepEP kernels only support dispatching block scales.
        # * For expert kernels that require unquantized inputs,
        #   defer quantization to FusedMoEExpertsPermuteUnpermute.
        if not quant_config.is_block_quantized and not defer_input_quant:
            # Quantize after dispatch.
            expert_x_scale = None
            if expert_x.numel() != 0:
                # TODO: support per_act_token_quant,
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # * DeepEP only supports fp8 block scales so quantize
        #   before the dispatch for these models.
        # * For all other quantization, dispatch after.
        # * For expert kernels that require unquantized inputs,
        #   defer quantization to FusedMoEExpertsPermuteUnpermute.
        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        assert handle is not None

        # fused_expert_output can have 0 tokens - This happens when none of the
        # tokens from the all2all reach this EP rank.
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        previous_event = dbo_get_previous_event(self.buffer.capture)
        dbo_yield_and_switch_from_compute_to_comm()
        assert fused_expert_output.dtype == torch.bfloat16, (
            f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
        )
        combined_x, _, event = self.buffer.combine(
            # HT combine only supports BF16
            x=fused_expert_output,
            handle=handle,
            topk_weights=None,
            config=self._get_combine_config(),
            previous_event=previous_event,
            async_finish=do_async and not dbo_enabled(),
            allocate_on_comm_stream=False,
        )

        self._sync_dbo_comm_if_needed()

        dbo_switch_to_compute()

        if do_async:

            def _receiver():
                if event.event is not None:
                    event.current_stream_wait()
                dbo_switch_to_comm()
                output.copy_(combined_x, non_blocking=True)

                # TODO(lucas): refactor the modular kernel so this will be
                # handled there
                dbo_yield_and_switch_from_comm_to_compute()

            return _receiver
        else:
            # TODO(lucas): support this case with the refactored modular kernel
            assert not dbo_enabled()
            output.copy_(combined_x, non_blocking=True)
            return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        receiver = self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            True,
        )
        assert receiver is not None
        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
