"""Per phase CUDA event timing for the EP MoE path, off unless enabled.

Set VLLM_EP_TIME_DUMP to an output prefix. Each rank writes
<prefix>.rank<N>.json at process exit. No synchronization happens in the hot
path: events are only recorded, and elapsed_time is read once at exit after a
device synchronize, so the measurement does not change the schedule it
measures.

Phases recorded: "route" (expert selection and metadata), "layer" (the whole
routed expert call), "dispatch" (the all2all dispatch), "combine" (the all2all
combine). Expert compute is layer minus dispatch minus combine.

Every call is kept in order rather than pooled, because the engine runs dummy
profile passes before the real prefill and pooling them together corrupts the
statistics. The reader chunks the ordered list by the MoE layer count to
recover per pass numbers.

Only valid under eager execution: events recorded inside a captured CUDA graph
do not carry per replay timings.
"""

import contextlib
import os

_ON = os.environ.get("VLLM_EP_TIME_DUMP", "")
_spans: list = []


class _Span:
    __slots__ = ("tag", "layer", "s", "e")

    def __init__(self, tag, layer, s, e):
        self.tag = tag
        self.layer = layer
        self.s = s
        self.e = e


class _Rec:
    """Context manager that brackets a region with two CUDA events."""

    __slots__ = ("tag", "layer", "s", "e")

    def __init__(self, tag, layer):
        self.tag = tag
        self.layer = layer

    def __enter__(self):
        import torch

        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
        self.s.record()
        return self

    def __exit__(self, *exc):
        self.e.record()
        _spans.append(_Span(self.tag, self.layer, self.s, self.e))
        return False


class _Null:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# The disabled-path no-op MUST be contextlib.nullcontext, not the _Null instance
# above. _Null is correct at runtime but Dynamo cannot enter a user-defined context
# manager, and under cudagraph_mode FULL_AND_PIECEWISE the model is traced with
# aot_compile / fullgraph_capture, which cannot absorb the resulting graph break.
# That killed all 8 ranks in profile_run via moe_runner.py's `with span(...)`
# regions. PIECEWISE only graph-breaks, which is why the EP campaign never saw it.
# nullcontext has a Dynamo variable tracker.
_NULL = contextlib.nullcontext()


_installed = False
_RSCTL = os.environ.get("VLLM_EP_RS_CONTROL", "") == "1"
_SYNCDEEP = os.environ.get("VLLM_EP_DEEPEP_SYNC", "") == "1"
_COMBBAR = os.environ.get("VLLM_EP_COMB_BARRIER", "") == "1"
_PHASE2 = os.environ.get("VLLM_EP_PHASE2", "") == "1"
_ctl: dict = {}


def _rs_controls(comm, t):
    """Two controls for the combine reduce-scatter, measured in place.

    The engine's combine costs 787.3 us/layer at 8192 tokens while the SAME
    reduce-scatter of the SAME 117.44 MB bf16 buffer costs 332.3 us on a bare
    8 rank process group. Something in the engine path, not the hardware, is
    paying 2.37x. These two controls separate the two candidate causes without
    guessing, both issued from inside the real forward, immediately before the
    real collective, on the same 8 GPUs:

      rs_ctl_tiny   a 64 row reduce_scatter on the SAME pynccl comm, issued
                    FIRST. Its transfer is ~4 KB, three orders of magnitude
                    below the real combine, so any large time it reports cannot
                    be bytes: it is the wait that the first collective at this
                    point in the layer absorbs, whether that is late rank
                    arrival or the expert GEMM tail still draining the stream.
      rs_ctl_pcp    the same pynccl comm this communicator already holds, but
                    on persistent pre-allocated contiguous scratch. If this is
                    fast, the cost is in the BUFFER the engine hands over
                    (allocation, contiguity, or the output torch.empty) and is
                    fixable in the caller.
      rs_ctl_world  torch's default process group, which at TP=1/PCP=8 spans
                    exactly the same 8 ranks but is a DIFFERENT NCCL
                    communicator. If this is fast while rs_ctl_pcp is slow, the
                    cost belongs to the communicator the EP path was given
                    (channel count, NVLS resources), not to the buffer.

    MEASURED at 8192 tokens: tiny/pcp/world/real ordering separates a wait from
    a transfer, because the first arm pays 786.5 us/layer and the later arms pay
    333-334 us/layer on identical bytes through the identical communicator.

    Symmetric across ranks, so no deadlock, and it touches only scratch, so the
    model output cannot change. It does add traffic, so wall time rises and the
    real collective's own number is read only relative to these controls.
    """
    import torch

    key = (tuple(t.shape), t.dtype)
    buf = _ctl.get(key)
    if buf is None:
        n = t.shape[0] // comm.world_size
        w = comm.world_size
        buf = (
            torch.empty_like(t.contiguous()),
            torch.empty((n,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device),
            torch.empty((8 * w, 8), dtype=t.dtype, device=t.device),
            torch.empty((8, 8), dtype=t.dtype, device=t.device),
        )
        _ctl[key] = buf
    ci, co, ti, to = buf
    pc = getattr(comm, "pynccl_comm", None)
    if pc is not None:
        with _Rec("rs_ctl_tiny", "moe"):
            pc.reduce_scatter(to, ti)
        with _Rec("rs_ctl_pcp", "moe"):
            pc.reduce_scatter(co, ci)
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() == comm.world_size:
            with _Rec("rs_ctl_world", "moe"):
                dist.reduce_scatter_tensor(co, ci)
    except Exception:
        pass


def _install_comm_probe():
    """Wrap the two collectives the EP path uses, at the communicator level.

    Done lazily on the first span so every module is already imported, and
    contained here so the tree needs no further edits. Records the MEASURED
    byte count and dtype with the time, plus whether the equal-size collapse
    that unlocks the symmetric-memory path actually happened, so the
    dispatch/combine cost gap is read off the wire rather than inferred.
    """
    global _installed
    if _installed:
        return
    _installed = True
    try:
        from vllm.distributed.device_communicators import cuda_communicator as CC
    except Exception:
        return
    cls = getattr(CC, "CudaCommunicator", None)
    if cls is None:
        return

    def mk(fn, nm):
        def g(self, input_, *a, **k):
            import torch

            t = input_
            if not isinstance(t, torch.Tensor):
                t = t[0] if isinstance(t, (list, tuple)) and t else None
            sizes = k.get("sizes")
            if sizes is None:
                for x in a:
                    if isinstance(x, list):
                        sizes = x
            nb = t.numel() * t.element_size() if t is not None else 0
            dt = str(t.dtype).replace("torch.", "") if t is not None else "?"
            eq = 1 if (sizes is None or len(set(sizes)) == 1) else 0
            try:
                symm = 1 if CC.should_nccl_symm_mem_ag_rs() else 0
            except Exception:
                symm = -1
            cg = 1 if (t is not None and t.is_contiguous()) else 0
            key = f"{nm}|nb={nb}|dt={dt}|eq={eq}|symmok={symm}|cg={cg}"
            if _RSCTL and nm == "reduce_scatterv" and t is not None:
                _rs_controls(self, t)
            with _Rec("comm", key):
                return fn(self, input_, *a, **k)

        g._ep_wrapped = True
        return g

    for nm in ("all_gatherv", "reduce_scatterv"):
        f = getattr(cls, nm, None)
        if f is not None and not getattr(f, "_ep_wrapped", False):
            setattr(cls, nm, mk(f, nm))


def _install_deepep_probe():
    """Time the DeepEP path at its own primitives, with no tree edits.

    The AgRs arm is timed through spans in naive_dp_ep.py plus a wrapper on the
    two communicator collectives. DeepEP uses neither: it takes
    modular_kernel's ASYNC prepare path and moves tokens inside its own kernels,
    so the existing spans record nothing but "route" and "layer" and the
    comparison has no phases in it.

    Both classes involved are plain Python, so wrapping their methods here keeps
    every edit in one file and keeps the two arms on the same tag names:

      layout    Buffer.get_dispatch_layout, the routing metadata DeepEP needs
      dispatch  Buffer.dispatch
      combine   Buffer.combine
      prep      DeepEPHTPrepareAndFinalize.prepare_async (nests dispatch)
      recv      the receiver returned by prepare_async, where an async dispatch
                is actually waited on
      fin       DeepEPHTPrepareAndFinalize.finalize_async (nests combine)

    With async_finish set, dispatch returns before its transfer is done and the
    wait lands in recv rather than in dispatch, which is the distinction the
    AgRs arm could not make.
    """
    pairs = []
    try:
        import deep_ep

        pairs += [
            (deep_ep.Buffer, "get_dispatch_layout", "layout"),
            (deep_ep.Buffer, "dispatch", "dispatch"),
            (deep_ep.Buffer, "combine", "combine"),
        ]
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.fused_moe.prepare_finalize import deepep_ht

        cls = getattr(deepep_ht, "DeepEPHTPrepareAndFinalize", None)
        if cls is not None:
            pairs += [
                (cls, "prepare_async", "prep"),
                (cls, "finalize_async", "fin"),
                (cls, "_receiver", "recv"),
            ]
    except Exception:
        pass

    def mk(fn, tag):
        def g(*a, **k):
            with _Rec(tag, "moe"):
                return fn(*a, **k)

        g._ep_wrapped = True
        return g

    if _SYNCDEEP:
        _make_deepep_sync()

    if _COMBBAR:
        pairs = [p for p in pairs if p[2] != "combine"]
        _install_combine_barrier()

    for obj, name, tag in pairs:
        f = getattr(obj, name, None)
        if f is None or getattr(f, "_ep_wrapped", False):
            continue
        # get_dispatch_layout and friends are plain methods; capture is a
        # staticmethod and is deliberately left alone.
        setattr(obj, name, mk(f, tag))


def _install_combine_barrier():
    """Split DeepEP's combine into arrival skew and real transfer cost.

    MEASURED at 8192 in the synchronous arm: `Buffer.combine` costs 1132.3
    us/layer against a bare 8 rank all-to-all combine floor of 191.0 us for the
    same bf16 volume, which is 5.93x. Dispatch over the same fabric in the same
    layer sits at 0.92x its own floor, so the fabric is not the explanation and
    the excess has to be attributed rather than guessed at.

    A tiny collective issued immediately before the combine IS a barrier: its
    transfer is ~4 KB, three orders of magnitude below the combine, so whatever
    time it reports is the spread in when the 8 ranks arrive at this point in the
    layer, not bytes. The combine that follows then starts from a synchronized
    state and reports its own cost with the wait removed:

      comb_barrier   arrival skew, i.e. the exposed expert GEMM imbalance
      combine        transfer plus receive side reduction, wait excluded

    The same trick already worked on the AgRs reduce-scatter, where a 4 KB
    control cost 495.0 us/layer and the real 117.44 MB collective behind it cost
    332.3 us. Symmetric across ranks so it cannot deadlock, and it touches only
    scratch so the model output cannot change. It adds traffic and a real
    synchronization, so wall time rises and this arm is read only for the split.
    """
    try:
        import deep_ep
        import torch
        import torch.distributed as dist
    except Exception:
        return
    f = getattr(deep_ep.Buffer, "combine", None)
    if f is None or getattr(f, "_ep_wrapped", False):
        return
    box = {}

    def g(*a, **k):
        t = box.get("t")
        if t is None and dist.is_initialized():
            t = box["t"] = torch.zeros(8, 8, dtype=torch.bfloat16, device="cuda")
        if t is not None:
            with _Rec("comb_barrier", "moe"):
                dist.all_reduce(t)
        with _Rec("combine", "moe"):
            return f(*a, **k)

    g._ep_wrapped = True
    deep_ep.Buffer.combine = g


def _make_deepep_sync():
    """Put the DeepEP transfers back on the compute stream so they can be timed.

    deepep_ht.py hardcodes self.async_prepare = True (line 62) and
    finalize_async passes do_async=True (line 425), so both dispatch and combine
    run with async_finish and complete on DeepEP's OWN comm stream. Events
    recorded on the compute stream then do not bracket them: combine reads 3.1
    us/layer and roughly 750 us reappears inside the expert span, which makes
    the phase split incomparable to the AgRs arm.

    supports_async() must stay True. Returning False routes modular_kernel to
    the fully synchronous path, which never launches the shared expert, and the
    run dies at shared_experts.py:175 on `assert self._output[self._output_idx]
    is not None` during the profile pass. So the async STRUCTURE is preserved and
    only the two async_finish flags are flipped:

      async_prepare = False   dispatch issues on the compute stream
      finalize_async          re-pointed at _finalize(..., do_async=False), which
                              runs combine plus the output copy inline and hands
                              modular_kernel a no op receiver so its contract
                              still holds

    This changes the schedule, so this arm is for the phase split only and its
    wall time is never the number quoted. The async arm owns that.
    """
    try:
        from vllm.model_executor.layers.fused_moe.prepare_finalize import deepep_ht
    except Exception:
        return
    cls = getattr(deepep_ht, "DeepEPHTPrepareAndFinalize", None)
    if cls is None:
        return
    base_init = cls.__init__
    if not getattr(base_init, "_ep_sync", False):

        def init(self, *a, **k):
            base_init(self, *a, **k)
            self.async_prepare = False

        init._ep_sync = True
        cls.__init__ = init

    if getattr(cls.finalize_async, "_ep_sync", False):
        return

    def finalize_async(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
        return lambda: None

    finalize_async._ep_sync = True
    cls.finalize_async = finalize_async


_hspans: list = []
_phase2_done = False
# Which collective launched last. Both exposed waits go through the same
# EventOverlap method, so the tag is decided by this rather than by guessing.
_WPH = ["d"]


class _HRec:
    """Host wall clock bracket, appended to a separate list.

    Not interchangeable with _Rec. _Rec measures DEVICE time between two CUDA
    events; this measures the CPU time the Python call itself occupies. The two
    differ exactly when a call blocks the host, which DeepEP's intranode
    dispatch does when it reads the per expert receive counts back.
    """

    __slots__ = ("tag", "t0")

    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        import time

        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        import time

        _hspans.append((self.tag, (time.perf_counter() - self.t0) * 1e3))
        return False


def _install_phase2_probe():
    """Split the MoE layer so nothing is left as a derived residual.

    Tags added, all keyed on layer name "moe" so the reader chunks them per pass
    exactly like the existing tags:

      dwait    EventOverlap.current_stream_wait for the DISPATCH event, i.e. the
               exposed dispatch wait on the compute stream
      gemm     FusedMoEModularKernel._fused_experts, the routed expert GEMMs
      wreduce  the TopKWeightAndReduce apply that runs inside _finalize BEFORE
               Buffer.combine, so it gates the combine launch
      shared   _maybe_apply_shared_experts, placed by the engine between the
               combine launch and the combine wait
      cwait    the same current_stream_wait for the COMBINE event
      cfin     the whole finalize receiver, so cfin minus cwait is the output copy
      hdisp    HOST wall time of Buffer.dispatch
      hcomb    HOST wall time of Buffer.combine
      hlayer   HOST wall time of the whole modular apply

    The fixed order inside one layer is dispatch, dwait, gemm, wreduce, combine,
    shared, cwait, which is what makes the single wait method separable.
    """
    global _phase2_done
    if _phase2_done:
        return
    _phase2_done = True

    # _ep_wrapped must be set. span() re-runs _install_deepep_probe on every
    # call (_installed is never assigned True), and that installer only skips a
    # method already carrying this attribute. Without it, Buffer.dispatch gains
    # one more nested device span per MoE layer per pass and the number grows
    # without bound.
    def dev(fn, tag):
        def g(*a, **k):
            with _Rec(tag, "moe"):
                return fn(*a, **k)

        g._ep_wrapped = True
        return g

    def host(fn, tag, phase):
        def g(*a, **k):
            with _HRec(tag):
                r = fn(*a, **k)
            _WPH[0] = phase
            return r

        g._ep_wrapped = True
        return g

    def shaped(fn, tag, phase):
        """host() plus the payload shape, so bytes and time come from one call.

        Achieved bandwidth is the only way to tell a collective that is at its
        floor from one that is not, and a byte count taken from a route dump is a
        different measurement than the one the timer brackets. `x` is a keyword
        in both call sites and can be a (tokens, scales) tuple for fp8 dispatch,
        in which case both parts are counted. Shapes are host side metadata, so
        reading them adds no synchronization.
        """

        def g(*a, **k):
            x = k.get("x")
            if x is None and len(a) > 1:
                x = a[1]
            nb = 0
            rows = 0
            for t in x if isinstance(x, tuple) else (x,):
                try:
                    nb += t.numel() * t.element_size()
                    rows = max(rows, t.shape[0])
                except Exception:
                    pass
            with _HRec(tag):
                r = fn(*a, **k)
            _WPH[0] = phase
            # Not "_rows": the reader's last_pass() already stores a
            # "<tag>_rows" key for its own per layer block list, and a trace tag
            # of that name would overwrite it.
            _hspans.append((tag + "_nrow", float(rows)))
            _hspans.append((tag + "_nkb", nb / 1024.0))
            return r

        g._ep_wrapped = True
        return g

    try:
        import deep_ep

        # Wrapped OUTSIDE the existing device span, so the host number includes
        # the same region the device number covers.
        deep_ep.Buffer.dispatch = shaped(deep_ep.Buffer.dispatch, "hdisp", "d")
        deep_ep.Buffer.combine = shaped(deep_ep.Buffer.combine, "hcomb", "c")

        csw = deep_ep.EventOverlap.current_stream_wait

        def wait(self, *a, **k):
            with _Rec("dwait" if _WPH[0] == "d" else "cwait", "moe"):
                return csw(self, *a, **k)

        deep_ep.EventOverlap.current_stream_wait = wait
    except Exception as e:  # pragma: no cover
        print(f"[ep_timer] phase2 deep_ep hook failed: {e!r}")

    try:
        from vllm.model_executor.layers.fused_moe import modular_kernel as _mk

        # NOT FusedMoEModularKernel. In this tree the class that owns
        # _prepare / _fused_experts / _finalize is FusedMoEKernelModularImpl;
        # FusedMoEKernel is the thin dispatcher above it.
        k = _mk.FusedMoEKernelModularImpl
        k._fused_experts = dev(k._fused_experts, "gemm")
        k._maybe_apply_shared_experts = dev(k._maybe_apply_shared_experts, "shared")

        # Host time of the whole call, for comparison against the device layer.
        _apply = k.apply

        def apply(*a, **kw):
            with _HRec("hlayer"):
                return _apply(*a, **kw)

        k.apply = apply
    except Exception as e:  # pragma: no cover
        print(f"[ep_timer] phase2 modular hook failed: {e!r}")

    try:
        from vllm.model_executor.layers.fused_moe import (
            topk_weight_and_reduce as twr,
        )

        # All four, because which one _finalize ends up calling depends on the
        # expert backend: the first run wrapped only two and recorded nothing,
        # which reads identically to "the reduce is free" and is not.
        for nm in (
            "TopKWeightAndReduceContiguous",
            "TopKWeightAndReduceNaiveBatched",
            "TopKWeightAndReduceNoOP",
            "TopKWeightAndReduceDelegate",
        ):
            c = getattr(twr, nm, None)
            if c is not None:
                c.apply = dev(c.apply, "wreduce")
    except Exception as e:  # pragma: no cover
        print(f"[ep_timer] phase2 reduce hook failed: {e!r}")

    try:
        from vllm.model_executor.layers.fused_moe.prepare_finalize import deepep_ht

        cls = deepep_ht.DeepEPHTPrepareAndFinalize
        fa = cls.finalize_async

        def finalize_async(*a, **k):
            r = fa(*a, **k)
            # HT returns a bare receiver, but tolerate the (hook, receiver) shape
            # so this does not silently stop recording if the interface changes.
            if isinstance(r, tuple):
                hk, rc = r
                return hk, dev(rc, "cfin")
            return dev(r, "cfin")

        finalize_async._ep_wrapped = True
        cls.finalize_async = finalize_async
    except Exception as e:  # pragma: no cover
        print(f"[ep_timer] phase2 finalize hook failed: {e!r}")


def span(tag, layer=""):
    """Bracket a region. Returns a no op unless timing is enabled."""
    if not _ON:
        return _NULL
    if not _installed:
        _install_comm_probe()
        _install_deepep_probe()
        if _PHASE2:
            _install_phase2_probe()
    return _Rec(tag, layer)


def _flush():
    if not _ON or not _spans:
        return
    import json

    import torch

    torch.cuda.synchronize()
    # ordered per call trace: {tag: [[layer, ms], ...]} in record order
    trace: dict = {}
    for sp in _spans:
        trace.setdefault(sp.tag, []).append(
            [str(sp.layer), round(sp.s.elapsed_time(sp.e), 4)]
        )
    # host timings carry no layer name, so they get the same "moe" key the
    # collective spans use and chunk per pass the same way
    for tag, ms in _hspans:
        trace.setdefault(tag, []).append(["moe", round(ms, 4)])
    # the process group is already torn down at atexit, so asking
    # get_ep_group() for the rank returns nothing and every worker then writes
    # the SAME filename and overwrites its peers. The device index is stable.
    try:
        rank = torch.cuda.current_device()
    except Exception:
        rank = -1
    with open(f"{_ON}.rank{rank}.json", "w") as f:
        json.dump({"trace": trace}, f)


if _ON:
    import atexit

    atexit.register(_flush)


# ---- matched whole-layer span ----
# begin()/end() exist so a region spanning many statements can be bracketed
# without being re-indented into a `with` block. Same _spans list and same
# _flush(), so the dump format and every existing reader are unchanged.
_open: list = []


def begin(tag, layer=""):
    """Open a span. Safe to call when timing is off (does nothing)."""
    if not _ON:
        return
    if not _installed:
        _install_comm_probe()
        _install_deepep_probe()
        if _PHASE2:
            _install_phase2_probe()
    import torch

    s = torch.cuda.Event(enable_timing=True)
    s.record()
    _open.append(_Span(tag, layer, s, None))


def end():
    """Close the most recently opened span.

    Unbalanced calls are ignored rather than raised, so a probe bug can never
    take the engine down mid-campaign.
    """
    if not _ON or not _open:
        return
    import torch

    sp = _open.pop()
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    sp.e = e
    _spans.append(sp)
