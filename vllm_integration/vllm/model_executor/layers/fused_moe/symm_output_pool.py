#!/usr/bin/env python3
"""A persistent symmetric output pool for the fused MoE epilogue.

WHAT THIS REPLACES. The harness allocated one symmetric buffer per token size and
rendezvoused each of them by hand:

    b = sm.empty(t * H_DIM, device=DEV, dtype=torch.bfloat16)
    h = sm.rendezvous(b, gname)

That is fine for a benchmark that knows its four shapes up front and never frees
anything, and unusable from an engine for three separate reasons:

  1. sm.rendezvous is a COLLECTIVE. It cannot run inside CUDA graph capture, and
     it cannot run on the per-request path at all, because a rank that reaches it
     while a peer does not deadlocks.
  2. A graph bakes the output pointer as a kernel argument. A per-call
     torch.empty hands back a different address on the next call, so a replay
     would reduce into freed memory. The buffer has to outlive every graph that
     captured it.
  3. Prefill token counts are not a small fixed set. Chunked prefill produces any
     M up to max_num_batched_tokens, so a shape-keyed table needs a miss path,
     and the miss path cannot rendezvous.

THE POLICY: ONE MAXIMUM-SIZE BUFFER PER SLOT, USED FROM OFFSET 0.

Rejected alternatives and why:

  a per-shape pool
      needs a rank-uniform shape table plus a miss path that cannot allocate, and
      buys nothing: the buffer is written [0, M*H) either way.
  one buffer with slots carved out at byte offsets
      makes every slot's alignment depend on the slot stride. 512-BYTE SHARD
      ALIGNMENT IS THE DOMINANT REDUCE LEVER (MEASURED), so alignment must not be
      a function of a size constant someone may later change. Separate
      allocations per slot keep every launch at offset 0 of its own allocation,
      where the symmetric allocator's own alignment applies.
  sizing to the largest captured shape
      prefill is not limited to captured shapes.

So each slot is its own sm.empty of max_tokens * hidden, and a launch of M tokens
uses the first M * hidden elements. Every shape therefore starts at the same
aligned base, and the 512-byte shard property that the reduce depends on holds
identically at 512 and at 16384 with no per-shape reasoning.

MEASURED on 8 x H200 before any of this was wired up, because both facts had to
hold for the policy to work at all:
  - every slot's base pointer AND its multicast pointer come back 512-byte
    aligned, so offset 0 is aligned for every shape (0x1002fc00a00 and
    0xdc4fc00a00, both %512 == 0; slot 1 likewise).
  - torch's multimem_all_reduce_ accepts a PREFIX of a larger symmetric
    allocation, at 512 / 2048 / 8192 / 16384 tokens, with correct values. So the
    reference collective runs on the same pool view the fused path writes and the
    comparison needs no second buffer.

HOW MANY SLOTS. One, unless the engine runs more than one microbatch in flight.
In this tree that is ParallelConfig.enable_dbo (default False) and
ParallelConfig.num_ubatches, which returns 2 when DBO is on. DBO alternates two
microbatches across a compute and a comm stream, so both can have an MoE output
live at once and one buffer would race. nslots is therefore not decoration: it
must equal num_ubatches, and the slot index must come from the ubatch id, which
is rank-uniform. Everything else in the serving path is single-flight: layer L's
MoE output is consumed by the residual add before layer L+1's MoE runs, on the
same stream, so stream ordering alone makes reuse safe without any host sync.

WHAT IS NOT IN HERE, DELIBERATELY. The barrier pad window and the graph-safe
sequence counter have exactly the same lifetime requirement as the output buffer,
so they are created here too and nowhere else. The counter is a plain device
tensor rather than symmetric memory because it is rank-local by design; see the
kernel comment on tp_seqctr. Nothing in this class is touched on the hot path:
pointers are read once at construction, views are cached, and there is no
allocation, no collective, and no synchronize after __init__ returns.
"""

import torch
import torch.distributed._symmetric_memory as sm


class SymmOutputPool:
    """Persistent symmetric output buffers plus the fused epilogue's coordination
    state. Construct once during initialization or warmup, never per call.
    """

    def __init__(
        self,
        group_name,
        max_tokens,
        hidden,
        pad_words,
        *,
        nslots=1,
        dtype=torch.bfloat16,
        device=None,
    ):
        assert nslots >= 1
        self.group_name = group_name
        self.max_tokens = int(max_tokens)
        self.hidden = int(hidden)
        self.nslots = int(nslots)
        self.dtype = dtype
        self.device = device or torch.cuda.current_device()

        n = self.max_tokens * self.hidden
        # Both the tensor and the handle are retained. Dropping the handle can
        # release the multicast mapping while a baked graph still points at it,
        # which is a use-after-free the kernel cannot detect.
        self._bufs, self._handles, self._mc, self._base = [], [], [], []
        for _ in range(self.nslots):
            b = sm.empty(n, device=self.device, dtype=dtype)
            h = sm.rendezvous(b, group_name)
            mc = h.multicast_ptr
            if not mc:
                raise RuntimeError(
                    "symmetric allocation has no multicast "
                    "pointer; the fused reduce needs one"
                )
            self._bufs.append(b)
            self._handles.append(h)
            self._mc.append(int(mc))
            self._base.append(int(b.data_ptr()))

        # pad_words comes from the binary's own prefill_wgmma_tp_pad_words at the
        # widest legal configuration, not from the formula rewritten here. A pool
        # that derives the size itself is a second source of truth and the two
        # drift the moment the kernel's pad layout changes.
        self.pad_words = int(pad_words)
        self._flags = sm.empty(self.pad_words, device=self.device, dtype=torch.int32)
        self._flags_handle = sm.rendezvous(self._flags, group_name)
        # Zeroed once. The pads are never cleared again: the graph-safe sequence
        # is monotone, so a stale pad value below the current sequence can only
        # make a barrier wait, never release early.
        self._flags.zero_()
        self.pads_ptr = int(self._flags_handle.buffer_ptrs_dev)

        # Persistent device sequence counter, outside every workspace because the
        # launcher memsets its workspace on every launch and a graph replays that
        # memset. Zeroed once here and never written from the host again.
        self._seqctr = torch.zeros(1, device=self.device, dtype=torch.int32)
        self.seqctr_ptr = int(self._seqctr.data_ptr())

        # Cached views, so the hot path does not even build a Python tensor.
        self._views = {}
        # Recorded for assert_stable(). A pointer that moves means something
        # freed a buffer a captured graph still refers to.
        self._fingerprint = (
            tuple(self._base),
            tuple(self._mc),
            self.pads_ptr,
            self.seqctr_ptr,
        )

    def output(self, num_tokens, slot=0):
        """The [num_tokens, hidden] view a launch writes. Offset 0, always."""
        key = (slot, num_tokens)
        v = self._views.get(key)
        if v is None:
            if num_tokens > self.max_tokens:
                raise ValueError(
                    f"{num_tokens} tokens exceeds pool max {self.max_tokens}"
                )
            v = self._bufs[slot][: num_tokens * self.hidden].view(
                num_tokens, self.hidden
            )
            self._views[key] = v
        return v

    def mc_ptr(self, slot=0):
        return self._mc[slot]

    def slot_buffer(self, slot=0):
        """The WHOLE slot allocation. Only a test needs this: it is how the range
        discipline is checked, by seeding the bytes past num_tokens * hidden and
        proving a launch leaves them alone."""
        return self._bufs[slot]

    def base_ptr(self, slot=0):
        return self._base[slot]

    def assert_stable(self):
        """Every pointer a graph may have baked is still the one we published."""
        now = (
            tuple(int(b.data_ptr()) for b in self._bufs),
            tuple(int(h.multicast_ptr) for h in self._handles),
            int(self._flags_handle.buffer_ptrs_dev),
            int(self._seqctr.data_ptr()),
        )
        if now != self._fingerprint:
            raise RuntimeError(f"pool pointers moved: {self._fingerprint} -> {now}")

    def bytes_per_slot(self):
        return self.max_tokens * self.hidden * self._bufs[0].element_size()

    def describe(self):
        return (
            f"slots={self.nslots} max_tokens={self.max_tokens} "
            f"hidden={self.hidden} bytes/slot={self.bytes_per_slot()} "
            f"pad_words={self.pad_words} "
            f"base0=0x{self._base[0]:x} mc0=0x{self._mc[0]:x} "
            f"base0%512={self._base[0] % 512} mc0%512={self._mc[0] % 512}"
        )
