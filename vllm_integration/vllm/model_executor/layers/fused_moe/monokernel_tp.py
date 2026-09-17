# SPDX-License-Identifier: Apache-2.0
"""Rank-uniform entry into the fused tensor-parallel MoE prefill path.

WHY THIS FILE EXISTS AT ALL. The fused path is not an optimization a rank can
opt into on its own. Its epilogue is a collective: communication blocks spin on
a barrier pad waiting for peers, and a rank that does not launch the fused
kernel never arrives. The peers then spin inside a kernel with no interrupt and
no timeout, which is a wedged GPU rather than an error. So the decision cannot
be per-rank, cannot be discovered lazily, and cannot be retried.

The existing entry point had exactly the shape that goes wrong:

    try:
        return torch.ops.vllm.moe_monokernel_prefill(...)
    except RuntimeError:
        logger.warning_once(...)
        return None

That is safe for the ordinary monokernel, whose only effect is local: a rank
that falls back to the modular path computes the same math and still joins the
same external all-reduce, so nothing hangs. It is unsafe the moment the kernel
itself communicates, because "this rank gave up" is invisible to the peers it
left waiting.

THE DECISION, AND WHY EACH INPUT IS RANK-UNIFORM.

  configuration          Identical by construction. VllmConfig is built once and
                         handed to every worker, so max_num_batched_tokens,
                         num_ubatches and hidden size agree. Read at quant
                         method construction and passed in, not re-read here.
  token count            TP replicates the batch, so every rank in the group
                         sees the same M for the same forward.
  the enable flag        An environment variable, so NOT uniform by
                         construction. It is therefore folded into the vote
                         below rather than checked before it.
  capability of the .so  Also not uniform by construction: each rank loads its
                         own file from its own environment, so one rank can hold
                         a binary without the fused entry point.
  pool availability      A collective itself (rendezvous), so it either succeeds
                         on every rank or fails on every rank, and it is folded
                         into the same vote.

Only tp_size is allowed to short-circuit ahead of the vote, because it comes
from the config and cannot disagree. Everything else reaches the MIN all-reduce,
which is what keeps a rank from returning early while its peers wait in a
collective. After the vote the per-call decision reads only the agreed flag and
M, and nothing left in it can differ between ranks.

AFTER ENTRY THERE IS NO FALLBACK, AND THAT IS THE SAFE CHOICE. Once the gate
says fused, every rank launches fused. A non-zero return code from the launch is
not converted into a local fallback: doing so would drop this rank out of a
collective its peers have already entered. It propagates, which fails the
engine. Failing loudly is strictly better than the alternative, because the
alternative is a hang with no diagnostic. Everything that CAN be checked is
checked before entry, at init, collectively.

SIZE POLICY IS AN EXACT-MATCH TABLE, DELIBERATELY. Only 8192 and 16384 were
measured as wins on DS3. 2048 was measured and is not a win. Nothing in between
was measured at all, and a range gate would quietly claim those sizes, so the
gate is a dict lookup: an unlisted token count takes the ordinary path. ncomm is
a plateau rather than a tuned optimum, so it is one fixed value inside the
validated band instead of a per-size fit to boot noise.
"""

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# MEASURED on 8 x H200, DS3 TP=8: the fused epilogue wins at 8192 and 16384 and
# does not win at 512 or 2048. Keys are exact token counts; anything absent
# takes the ordinary path. Values are the chunk count C the collective is
# released in.
_FUSED_CHUNKS_BY_TOKENS: dict[int, int] = {
    8192: 4,
    16384: 8,
}

# Communication blocks carved out of the persistent grid. The 12..16 band was
# measured as a plateau, so this is a fixed value inside it rather than a
# per-size choice fitted to run-to-run variation.
_FUSED_NCOMM = 14


def _env_chunk_table(default: dict[int, int]) -> dict[int, int]:
    """Optional override of the chunk table, for tuning a new model geometry.

    Unset returns the MEASURED DS3 table unchanged. A bare integer applies to
    every token count already in the table; a "T:C,T:C" list replaces it. A
    malformed value raises, because a tuning knob that silently does nothing
    would make a null result look like a property of the geometry.
    """
    raw = os.environ.get("VLLM_PREFILL_MONOKERNEL_TP_CHUNKS", "").strip()
    if not raw:
        return dict(default)
    if ":" not in raw:
        c = int(raw)
        if c < 1:
            raise ValueError("VLLM_PREFILL_MONOKERNEL_TP_CHUNKS must be >= 1")
        return {t: c for t in default}
    out: dict[int, int] = {}
    for item in raw.split(","):
        t, c = item.split(":")
        out[int(t)] = int(c)
    if not out or any(c < 1 for c in out.values()):
        raise ValueError(
            f"VLLM_PREFILL_MONOKERNEL_TP_CHUNKS {raw!r} is not a usable T:C list"
        )
    return out


def _env_ncomm(default: int) -> int:
    """Optional override of the communication-block count. Unset keeps 14."""
    raw = os.environ.get("VLLM_PREFILL_MONOKERNEL_TP_NCOMM", "").strip()
    if not raw:
        return default
    n = int(raw)
    if n < 1:
        raise ValueError("VLLM_PREFILL_MONOKERNEL_TP_NCOMM must be >= 1")
    return n


# Folding the shared expert output into the same collective. On by default when
# the loaded binary can do it, because a fused epilogue that reduces only the
# routed half does not remove a collective from this model, it ADDS one: the
# runner then reduces the shared half on its own, at identical bytes. MEASURED,
# that cost 5.53% at 8192 and 3.54% at 16384 on DS3 TP=8. Set to 0 for the
# same-binary control that separates the residual read from the fold.
_FOLD_SHARED = os.environ.get("VLLM_PREFILL_MONOKERNEL_TP_FOLD_SHARED", "1") not in (
    "",
    "0",
)


class _FusedTPPlan:
    """The agreed fused-path decision for this process, plus the buffers it needs.

    Constructed once, on the first query, at which point the TP process group is
    live and the vote can be taken. Everything the hot path reads is a plain
    attribute set here.
    """

    def __init__(
        self, hidden: int, max_tokens: int, nslots: int, enabled: bool
    ) -> None:
        from vllm.distributed.parallel_state import get_tp_group

        self.usable = False
        # Whether the epilogue carries the shared expert output too. Decided by
        # the SAME vote as usable: a rank that folded while a peer did not would
        # produce a silently different answer on that rank, which is worse than
        # a refusal, so it is never a local decision.
        self.fold_shared = False
        self._local_fold = False
        self.reason = ""
        self.pool = None
        self.chunks_by_tokens: dict[int, int] = {}
        self.ncomm = _env_ncomm(_FUSED_NCOMM)
        self.rank = 0
        self.world = 1
        self.nslots = max(1, int(nslots))
        self.max_tokens = int(max_tokens)
        self.hidden = int(hidden)
        self._caps: dict = {}

        # tp_size comes from the config, so it is the one input that may decide
        # this ahead of the vote: it cannot disagree between ranks.
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            self.reason = "torch.distributed not initialized"
            return
        tp = get_tp_group()
        self.rank = tp.rank_in_group
        self.world = tp.world_size
        if self.world < 2:
            self.reason = f"tp_size={self.world}"
            return

        self.chunks_by_tokens = {
            t: c
            for t, c in _env_chunk_table(_FUSED_CHUNKS_BY_TOKENS).items()
            if t <= self.max_tokens
        }

        local_ok, local_reason = self._local_capability(enabled)
        # Every rank reaches this. A rank that returned here instead would leave
        # its peers inside the all-reduce.
        if not self._vote(local_ok, tp):
            self.reason = (
                local_reason if not local_ok else "a peer rank could not run fused"
            )
            logger.info(
                "fused TP MoE prefill disabled: %s (local_capable=%s)",
                self.reason,
                local_ok,
            )
            return

        self.usable = True
        logger.info(
            "fused TP MoE prefill enabled: rank=%d world=%d slots=%d "
            "max_tokens=%d hidden=%d ncomm=%d sizes=%s chunks=%s "
            "fold_shared=%s",
            self.rank,
            self.world,
            self.nslots,
            self.max_tokens,
            self.hidden,
            self.ncomm,
            sorted(self.chunks_by_tokens),
            dict(sorted(self.chunks_by_tokens.items())),
            self.fold_shared,
        )

    def _local_capability(self, enabled: bool) -> tuple[bool, str]:
        import vllm._custom_ops as ops

        if not enabled:
            return False, "VLLM_PREFILL_MONOKERNEL_TP_FUSE is not set"
        if not current_platform.is_cuda():
            return False, "not CUDA"
        if not self.chunks_by_tokens:
            return False, (
                f"no measured fused size fits max_num_batched_tokens={self.max_tokens}"
            )
        try:
            caps = ops.prefill_monokernel_tp_caps()
        except Exception as e:  # noqa: BLE001 - a load failure is a capability answer
            return False, f"monokernel .so did not load: {e}"
        if not caps.get("usable"):
            return False, f"monokernel .so is not fused-TP capable: {caps}"
        # The device sequence counter is what lets a replayed graph advance the
        # barrier sequence. Without it the launch takes a host value, which a
        # graph bakes and replays unchanged, so a second replay would wait on a
        # sequence that has already passed. Refuse rather than ship that.
        if not caps.get("tp_has_devseq"):
            return False, "monokernel .so has no device sequence counter"
        want = max(self.chunks_by_tokens.values())
        if int(caps["tp_max_c"]) < want:
            return False, (
                f"binary tp_max_c={caps['tp_max_c']} below required chunks={want}"
            )
        if self.ncomm >= int(caps["grid_size"]):
            return False, f"ncomm={self.ncomm} not below grid_size={caps['grid_size']}"
        self._caps = caps
        # Not a requirement for the fused path, so it does not refuse here: a
        # binary without the residual entry point still runs the routed-only
        # epilogue. It is voted on separately so that the fold itself is
        # rank-uniform.
        self._local_fold = bool(_FOLD_SHARED and caps.get("has_tp_residual"))
        return True, ""

    def _vote(self, local_ok: bool, tp) -> bool:
        """MIN all-reduce over the TP group: fused runs only if every rank can.

        This is the only collective the decision performs, it happens once, and
        it happens here rather than on the per-call path precisely so that no
        later call has to ask a peer anything.
        """
        # Two questions, one collective: may the fused path run at all, and
        # does every rank's binary also fold the shared half. MIN on both, so
        # either is refused by a single dissenting rank.
        v = torch.tensor(
            [1 if local_ok else 0, 1 if self._local_fold else 0],
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(
            v, op=torch.distributed.ReduceOp.MIN, group=tp.device_group
        )
        agreed = v.tolist()
        if int(agreed[0]) != 1:
            return False
        self.fold_shared = bool(int(agreed[1]))
        return self._build_pool(tp)

    def _build_pool(self, tp) -> bool:
        """Rendezvous the persistent pool once, here, and nowhere else.

        Every rank reaches this line or none does: it is only called after the
        vote passed on all of them, so the rendezvous inside cannot deadlock on
        a missing peer.
        """
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.fused_moe.symm_output_pool import (
            SymmOutputPool,
        )

        if torch.cuda.is_current_stream_capturing():
            # rendezvous is a collective and cannot be captured. Capture state
            # is itself rank-uniform, so refusing here refuses on every rank.
            logger.warning(
                "fused TP MoE prefill disabled: the pool cannot be created "
                "during CUDA graph capture, and no eager call reached it first"
            )
            return False
        caps = self._caps
        pad_words = ops.prefill_monokernel_tp_pad_words(
            int(caps["tp_max_c"]), int(caps["grid_size"]) - 1, self.world
        )
        try:
            self.pool = SymmOutputPool(
                tp.device_group.group_name,
                self.max_tokens,
                self.hidden,
                pad_words,
                nslots=self.nslots,
                device=torch.cuda.current_device(),
            )
        except Exception as e:  # noqa: BLE001
            # A rendezvous failure is symmetric: it fails on every rank in the
            # group, so returning False here returns False everywhere.
            logger.warning("fused TP MoE prefill disabled: pool failed: %s", e)
            return False
        logger.info("fused TP MoE prefill pool: %s", self.pool.describe())
        return True

    def chunks_for(self, num_tokens: int) -> int:
        return self.chunks_by_tokens.get(int(num_tokens), 0)

    def slot(self) -> int:
        """The pool slot for the microbatch currently executing.

        Rank-uniform because the ubatch index comes from the same scheduler
        decision on every rank. Not hardcoded to 0 because DBO runs two
        microbatches concurrently and both can hold an MoE output live.
        """
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        s = int(dbo_current_ubatch_id())
        if not 0 <= s < self.nslots:
            raise RuntimeError(
                f"ubatch id {s} outside pool slots [0, {self.nslots}); the pool "
                f"was sized from ParallelConfig.num_ubatches"
            )
        return s


_plan: _FusedTPPlan | None = None

# Token counts already reported. The engine's prefill width is the measured
# prompt total, not a round number, and the gate is an exact match, so which
# widths reached this is the difference between a fused arm and a fused label.
_seen_sizes: set[int] = set()

# Diagnostic only, off unless asked for. The deduplicated log above says WHICH
# widths reached the gate; it cannot say how many times or in what order, which
# is what a per-iteration latency split needs. Environment rather than a config
# field because it is a measurement aid and must not reach a serving default.
_LOG_ALL = os.environ.get("MONO_TP_LOG_ALL", "0") not in ("", "0")
_LOG_ALL_MIN = int(os.environ.get("MONO_TP_LOG_MIN", "512"))
_call_seq = 0


def fused_tp_plan(hidden: int, max_tokens: int, nslots: int) -> _FusedTPPlan:
    global _plan
    if _plan is None:
        _plan = _FusedTPPlan(
            hidden, max_tokens, nslots, envs.VLLM_PREFILL_MONOKERNEL_TP_FUSE
        )
    return _plan


def reset_fused_tp_plan() -> None:
    """Drop the cached plan, releasing the pool with it. For tests that build
    more than one configuration in a single process."""
    global _plan
    _plan = None
    _seen_sizes.clear()


def output_is_fused(out: torch.Tensor | None) -> bool:
    """Was this tensor produced by the fused epilogue, and therefore already
    summed across the TP group?

    Answered from the tensor's address rather than from a flag, because a flag
    would have to be mutable process state and this is read from inside an
    opaque op that a CUDA graph may replay. The fused output is a view of a
    symmetric pool slot: a separate allocation that the caching allocator never
    hands out, whose address is fixed for the life of the process and baked
    unchanged into any captured graph. So an address match is proof, and a
    non-fused output cannot collide with it.
    """
    p = _plan
    if out is None or p is None or not p.usable or p.pool is None:
        return False
    return out.data_ptr() == p.pool.base_ptr(p.slot())


def _scales_f32c(layer, attr: str) -> torch.Tensor:
    """The scale tensor as fp32 contiguous, converted at most once per layer.

    The op converts its scale arguments on every call, and the weights are
    frozen after loading, so the conversion is a constant being recomputed. The
    harness already removes it for the allocating op (wrapfix's S, bitwise
    identical, 9.5 us a call where it was measured); the fused path takes the
    mutating op, which wrapfix does not wrap. Caching here is what makes the
    fused arm and the ordinary monokernel arm the same condition apart from the
    epilogue.

    Free when the scales are already fp32 and contiguous: .float() and
    .contiguous() both return self, so the cache holds the original tensor and
    costs no memory.
    """
    t = getattr(layer, attr)
    key = f"_mk_f32_{attr}"
    c = getattr(layer, key, None)
    if c is None or c.shape != t.shape or c.device != t.device:
        c = t.float().contiguous()
        setattr(layer, key, c)
    return c


def fused_tp_launch(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    layer,
    weight_scale_name: str,
    bias: torch.Tensor | None,
    max_tokens: int,
    nslots: int,
    residual: torch.Tensor | None = None,
    out_scale: float = 1.0,
) -> torch.Tensor | None:
    """Run the fused TP MoE prefill, or return None to take the ordinary path.

    Returning None is a rank-uniform outcome: every input to that decision is
    either configuration, the replicated token count, or capability agreed at
    init. Once this function launches, it does not fall back.
    """
    hidden = layer.w2_weight.size(1)
    plan = fused_tp_plan(hidden, max_tokens, nslots)
    if not plan.usable:
        return None
    chunks = plan.chunks_for(x.size(0))
    n = int(x.size(0))
    if _LOG_ALL and n >= _LOG_ALL_MIN:
        global _call_seq
        _call_seq += 1
        logger.info(
            "fused TP MoE prefill call #%d tokens=%d chunks=%d",
            _call_seq,
            n,
            chunks,
        )
    if n not in _seen_sizes:
        _seen_sizes.add(n)
        logger.info(
            "fused TP MoE prefill sees tokens=%d -> %s",
            n,
            f"FUSED chunks={chunks} ncomm={plan.ncomm}"
            if chunks
            else "ordinary path, this token count is not a measured size",
        )
    if chunks == 0:
        return None
    if plan.fold_shared and residual is None:
        # The fold is the whole reason this epilogue removes a collective, so
        # entering it without the shared half would reinstate the separate
        # reduce this path exists to delete. Rank-uniform: residual availability
        # follows the shared-experts order, which follows the token count, which
        # TP replicates.
        logger.warning_once(
            "fused TP MoE prefill declined at tokens=%d: the fold is enabled "
            "but the shared expert output was not available before the launch",
            n,
        )
        return None
    if residual is not None:
        if residual.shape != (n, hidden) or residual.dtype != torch.bfloat16:
            raise RuntimeError(
                f"folded residual must be [{n}, {hidden}] bfloat16, got "
                f"{tuple(residual.shape)} {residual.dtype}"
            )
        if not residual.is_contiguous():
            raise RuntimeError("folded residual must be contiguous")

    slot = plan.slot()
    pool = plan.pool
    out = pool.output(x.size(0), slot)
    # Cheap and worth it: this is the check that catches a released multicast
    # mapping that a captured graph still points at, which is otherwise a
    # use-after-free the kernel cannot detect.
    pool.assert_stable()

    torch.ops.vllm.moe_monokernel_prefill_out(
        x,
        router_logits,
        layer.w13_weight,
        _scales_f32c(layer, f"w13_{weight_scale_name}"),
        layer.w2_weight,
        _scales_f32c(layer, f"w2_{weight_scale_name}"),
        out,
        layer.top_k,
        getattr(layer, "renormalize", True),
        bias,
        pool.mc_ptr(slot),
        pool.pads_ptr,
        pool.seqctr_ptr,
        plan.rank,
        plan.world,
        chunks,
        plan.ncomm,
        residual if plan.fold_shared else None,
        float(out_scale) if plan.fold_shared else 1.0,
    )
    return out
