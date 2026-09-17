#!/usr/bin/env python3
"""Correctness gate for the pad-aware silu kernel, on the fused_experts OUTPUT.

This is the gate for the shipped DeepSeek-V3 EP path, not for the persistent kernel.
It covers the pad-aware fused activation (`VLLM_EP_FAST_SILU`) on both the ragged and
the full expert GEMM grid (`VLLM_EP_MASKED_SUM`).

WHY THIS AND NOT THE KERNEL OUTPUT. A separate arms harness already proved the kernel
bitwise identical on the rows this rank owns. That is not sufficient to ship, because
the masked arm leaves the other rows UNWRITTEN, and whether that is safe is a property
of the second GEMM and the reducer, not of the silu kernel. So this gate compares the
thing the engine actually consumes: the bf16 output of the whole apply composition,
quantize through moe_sum, bitwise, on BOTH grids.

EVERY "UNWRITTEN IS FINE" CLAIM NEEDS AN ANTI-VACUITY COMPANION. A gate that poisons
a buffer nothing was going to read passes for the wrong reason. So part 2 runs the raw
kernel into a POISONED intermediate and checks two things that must BOTH hold:

  poison the rows the kernel SKIPPED   -> output must be UNCHANGED (they are unread)
  poison the rows the kernel WROTE     -> output must CHANGE      (they are read)

The second direction is what makes the first mean something. Without it, "unchanged"
would also be the answer if the second GEMM read nothing at all, or if the poison were
too small to survive the reduction.

THE POISON IS CHOSEN TO BE UNMISSABLE, NOT SUBTLE. fp8 e4m3 0x7E is +448, the format
maximum, and the paired scale is 1e30, so any row that is read contributes about 4.5e32
and lands as inf in the bf16 output. A poison inside the normal value range could be
absorbed by the reduction and would make a false pass possible.

NO HIDDEN HOST SYNC IS MEASURED, NOT INSPECTED. The silu call runs inside
torch.cuda.set_sync_debug_mode("error"), which raises on any device to host
synchronization, so a stray .item() in the new path fails the gate rather than showing up
later as a pipeline drain in the engine. That is scoped to the silu call alone, because
the surrounding harness legitimately syncs.

BOTH GRIDS ARE REQUIRED AND THEY ARE SAFE FOR DIFFERENT REASONS. On the ragged grid
(ignore_invalid_experts=True) sorted_token_ids holds owned slots only, so gemm2 never
indexes a skipped row. On the full rectangle the skipped rows ARE in sorted_token_ids,
but their expert_ids entry is -1 and fused_moe_kernel takes the off_experts == -1 branch
at fused_moe.py:167 and :424, which writes zeros and returns before loading A or the A
scales. Two different arguments, so two different gates.
"""
import os
import sys

# TREE is the vLLM source tree, as in the benchmark scripts. Unset means vLLM is
# already importable.
_TREE = os.environ.get("TREE", "")
if _TREE:
    sys.path.insert(0, _TREE)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.fused_moe import override_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fast_silu_quant import (  # noqa: E402
    _silu_mul_block_quant_kernel,
    _supported,
    silu_and_mul_per_block_quant_pad_aware,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: E402
    get_moe_configs,
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import (  # noqa: E402
    _resize_cache,
    moe_kernel_quantize_input,
)

HIDDEN = 7168
INTER = 2048
N2 = 2 * INTER
GLOBAL_E = 256
LOCAL_E = 32
BLOCK = [128, 128]
FP8 = torch.float8_e4m3fn
RANK = 0
FAKE = GLOBAL_E - 1
OWNED_MEAN = 2.0
POISON_FP8 = 0x7E          # +448, the e4m3 maximum
POISON_SCALE = 1.0e30

# smaller than the operating points on purpose: this is a numerics gate, and every
# point is run on both grids with three silu arms plus four poison variants
POINTS = [("T=1024 w8", 503, 8), ("T=8192 w8", 4019, 8), ("T=8192 w4", 4019, 4)]


def emap():
    m = torch.full((GLOBAL_E,), -1, dtype=torch.int32, device="cuda")
    m[RANK * LOCAL_E:(RANK + 1) * LOCAL_E] = torch.arange(
        LOCAL_E, dtype=torch.int32, device="cuda")
    return m


def weights():
    torch.manual_seed(11)
    w1 = torch.randn(LOCAL_E, N2, HIDDEN, device="cuda", dtype=torch.bfloat16).to(FP8)
    w2 = torch.randn(LOCAL_E, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16).to(FP8)
    w1s = torch.rand(LOCAL_E, N2 // 128, HIDDEN // 128, device="cuda",
                     dtype=torch.float32) * 0.02 + 0.01
    w2s = torch.rand(LOCAL_E, HIDDEN // 128, INTER // 128, device="cuda",
                     dtype=torch.float32) * 0.02 + 0.01
    fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=BLOCK)
    return w1, w2, w1s, w2s


def route(n, width, seed=1234):
    g = torch.Generator().manual_seed(seed)
    cnt = torch.randint(1, width + 1, (n,), generator=g)
    cnt = torch.clamp((cnt.float() * (OWNED_MEAN * n / float(cnt.sum()))).round()
                      .long(), 1, width)
    col = torch.arange(width).view(1, -1)
    keep = col < cnt.view(-1, 1)
    flat = torch.randint(0, LOCAL_E, (n, width), generator=g, dtype=torch.int32)
    ids = torch.where(keep, flat + RANK * LOCAL_E,
                      torch.full((n, width), FAKE, dtype=torch.int32))
    wts = torch.where(keep, torch.ones(n, width), torch.zeros(n, width))
    return (ids.cuda(), wts.cuda().to(torch.float32), keep.cuda().reshape(-1),
            int(keep.sum()))


def table_cfg(m):
    cfgs = get_moe_configs(LOCAL_E, INTER, "fp8_w8a8", BLOCK[0], BLOCK[1])
    keys = sorted(int(k) for k in cfgs if str(k).isdigit())
    cfg = dict(cfgs[min(keys, key=lambda k: abs(k - m))])
    cfg.pop("triton_version", None)
    return cfg


class Chain:
    """apply's composition, with the silu step swappable and the intermediate exposed."""

    def __init__(self, W, n, width, ragged):
        self.n, self.width, self.ragged = n, width, ragged
        self.w1, self.w2, self.w1s, self.w2s = W
        self.cfg = table_cfg(n)
        self.em = emap()
        self.ids, self.wts, self.keep, self.owned = route(n, width)
        torch.manual_seed(4242)
        self.a = torch.randn(n, HIDDEN, device="cuda", dtype=torch.bfloat16)
        self.ws2 = torch.empty(n * width * max(N2, HIDDEN), device="cuda",
                               dtype=torch.bfloat16)
        self.out = torch.empty(n, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def free(self):
        for k in ("a", "ws2", "out", "ids", "wts", "keep"):
            setattr(self, k, None)
        torch.cuda.empty_cache()

    def _front(self):
        """quantize, align, gemm1. Returns cache1 as a [rows, N2] view."""
        aq, asc = moe_kernel_quantize_input(self.a, None, FP8, False, BLOCK)
        s, e, npost = moe_align_block_size(
            self.ids, self.cfg["BLOCK_SIZE_M"], GLOBAL_E, self.em,
            ignore_invalid_experts=self.ragged)
        c1 = _resize_cache(self.ws2, (self.n, self.width, N2))
        invoke_fused_moe_triton_kernel(
            aq, self.w1, c1, asc, self.w1s, None, s, e, npost, False, self.width,
            self.cfg, compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
            block_shape=BLOCK)
        return c1.view(-1, N2), (s, e, npost)

    def _back(self, q, qs, grid):
        """gemm2 and the reducer. cache3 aliases workspace2, exactly as apply does."""
        s, e, npost = grid
        c3 = _resize_cache(self.ws2, (self.n, self.width, HIDDEN))
        invoke_fused_moe_triton_kernel(
            q, self.w2, c3, qs, self.w2s, self.wts, s, e, npost, True, 1, self.cfg,
            compute_type=tl.bfloat16, use_fp8_w8a8=True, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
            block_shape=BLOCK)
        if self.ragged:
            ops.moe_sum(c3, self.out, self.ids, self.em)
        else:
            ops.moe_sum(c3, self.out)
        return self.out.clone()

    def run(self, arm):
        """arm in {c, t1, t2}: the C kernel, the Triton kernel, the masked Triton kernel."""
        with override_config(self.cfg):
            c1, grid = self._front()
            if arm == "c":
                q, qs = ops.silu_and_mul_per_block_quant(c1, group_size=128,
                                                         quant_dtype=FP8)
            else:
                masked = arm == "t2"
                # no hidden host sync is MEASURED here, not inspected
                torch.cuda.set_sync_debug_mode("error")
                try:
                    q, qs = silu_and_mul_per_block_quant_pad_aware(
                        c1, group_size=128, quant_dtype=FP8,
                        topk_ids=self.ids if masked else None,
                        expert_map=self.em if masked else None)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            return self._back(q, qs, grid)

    def run_poisoned(self, which):
        """Masked kernel into a poisoned buffer.

        which='skipped' poisons only the rows the kernel does not write, which must not
        reach the output. which='written' poisons only the rows it does write, which must
        reach it. Both use the same poison, so a difference in the verdict cannot be a
        difference in poison strength.
        """
        rows = self.n * self.width
        ngroup = INTER // 128
        with override_config(self.cfg):
            c1, grid = self._front()
            q = torch.empty(rows, INTER, device="cuda", dtype=FP8)
            qs = torch.empty(rows, ngroup, device="cuda", dtype=torch.float32)
            q.view(torch.uint8).fill_(POISON_FP8)
            qs.fill_(POISON_SCALE)
            if which == "skipped":
                # poison first, then let the kernel overwrite the rows it owns
                _silu_mul_block_quant_kernel[(rows,)](
                    c1, q, qs, self.ids.reshape(-1), self.em, rows, H=INTER, G=128,
                    NG=ngroup, QMAX=448.0, MIN_SCALE=1.0 / (448.0 * 512.0), MASKED=True,
                    num_warps=4)
            else:
                # let the kernel write, THEN poison exactly the rows it wrote
                q.view(torch.uint8).zero_()
                qs.zero_()
                _silu_mul_block_quant_kernel[(rows,)](
                    c1, q, qs, self.ids.reshape(-1), self.em, rows, H=INTER, G=128,
                    NG=ngroup, QMAX=448.0, MIN_SCALE=1.0 / (448.0 * 512.0), MASKED=True,
                    num_warps=4)
                sel = self.keep
                q.view(torch.uint8)[sel] = POISON_FP8
                qs[sel] = POISON_SCALE
            return self._back(q, qs, grid)


def bitwise(a, b):
    """bf16 equality on the bit pattern, so a NaN or a signed zero cannot pass."""
    if a.shape != b.shape:
        return False
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


def part1(W):
    print()
    print("=== GATE PART 1: bitwise identity of the fused_experts OUTPUT (bf16 "
          "[recv, 7168], compared on the bit pattern) against the C silu path, DS3 rank "
          "local shape E_local=32 hidden=7168 inter=2048 fp8 w8a8 block 128x128, "
          "1 x H200. t1 is the Triton kernel over every row, t2 is the masked kernel.")
    print("%-12s %-8s %9s %9s %9s %9s %9s"
          % ("point", "ragged", "rows", "owned", "padded", "t1 bitwise", "t2 bitwise"))
    ok_all = True
    for nm, n, width in POINTS:
        for ragged in (False, True):
            ch = Chain(W, n, width, ragged)
            with override_config(ch.cfg):
                _c1, g = ch._front()
                npad = int(g[2].item())
            ref = ch.run("c")
            o1 = ch.run("t1")
            o2 = ch.run("t2")
            b1, b2 = bitwise(ref, o1), bitwise(ref, o2)
            ok_all = ok_all and b1 and b2
            print("%-12s %-8s %9d %9d %9d %9s %9s"
                  % (nm, ragged, n * width, ch.owned, npad, b1, b2))
            for lab, o in (("t1", o1), ("t2", o2)):
                if bitwise(ref, o):
                    continue
                d = (ref.float() - o.float()).abs()
                print("    %s differs: %d of %d elements, max abs dev %.6g, "
                      "ref max %.6g" % (lab, int((d > 0).sum()), d.numel(),
                                        float(d.max()), float(ref.float().abs().max())))
            del ref, o1, o2
            ch.free()
    return ok_all


def part2(W):
    print()
    print("=== GATE PART 2: anti-vacuity. Same poison (fp8 0x7E = +448 with scale 1e30, "
          "about 4.5e32 per contributing row) applied to the rows the masked kernel "
          "SKIPS and to the rows it WRITES. Skipped must leave the output bitwise "
          "unchanged; written must change it. Both directions are required, because "
          "'unchanged' alone is also what a gate that reads nothing would report.")
    print("%-12s %-8s %9s %9s %20s %20s"
          % ("point", "ragged", "rows", "owned", "skipped: unchanged", "written: changed"))
    ok_all = True
    for nm, n, width in POINTS:
        for ragged in (False, True):
            ch = Chain(W, n, width, ragged)
            ref = ch.run("c")
            o_sk = ch.run_poisoned("skipped")
            o_wr = ch.run_poisoned("written")
            unchanged = bitwise(ref, o_sk)
            changed = not bitwise(ref, o_wr)
            ok_all = ok_all and unchanged and changed
            print("%-12s %-8s %9d %9d %20s %20s"
                  % (nm, ragged, n * width, ch.owned, unchanged, changed))
            if not unchanged:
                d = (ref.float() - o_sk.float()).abs()
                print("    poisoning SKIPPED rows reached the output: %d of %d elements "
                      "differ, max abs dev %.6g. The unwritten rows are NOT unread on "
                      "this grid." % (int((d > 0).sum()), d.numel(), float(d.max())))
            if not changed:
                print("    poisoning WRITTEN rows did NOT reach the output, so this gate "
                      "is VACUOUS on this point and its 'unchanged' verdict says "
                      "nothing.")
            del ref, o_sk, o_wr
            ch.free()
    return ok_all


def part3():
    print()
    print("=== GATE PART 3: the wrapper's support guard and fallback. An unsupported "
          "shape must return the C kernel's own result bitwise, not raise and not "
          "silently produce something else.")
    print("%-28s %8s %10s %12s %12s" % ("shape", "H", "supported", "took", "bitwise"))
    cases = [((64, 2 * 2048), True), ((64, 2 * 1536), False), ((64, 2 * 4096), True),
             ((64, 2 * 1024), True), ((0, 2 * 2048), True)]
    ok_all = True
    for shape, want in cases:
        rows, gu = shape
        h = gu // 2
        x = torch.randn(max(rows, 1), gu, device="cuda", dtype=torch.bfloat16)
        if rows == 0:
            x = x[:0]
        got = _supported(x, 128, FP8)
        qa, sa = ops.silu_and_mul_per_block_quant(x, group_size=128, quant_dtype=FP8)
        qb, sb = silu_and_mul_per_block_quant_pad_aware(x, 128, FP8)
        bw = (qa.shape == qb.shape
              and bool(torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)))
              and sa.numel() == sb.numel()
              and bool(torch.equal(sa.reshape(-1), sb.reshape(-1))))
        ok = (got == want) and bw
        ok_all = ok_all and ok
        print("%-28s %8d %10s %12s %12s"
              % (str(tuple(x.shape)), h, got, "triton" if got else "C fallback", bw))
        if got != want:
            print("    support verdict %s but this shape was expected to be %s"
                  % (got, want))
        del x, qa, sa, qb, sb
        torch.cuda.empty_cache()
    return ok_all


def main():
    print("GPU %s, triton %s" % (torch.cuda.get_device_name(0), triton.__version__))
    W = weights()
    p1 = part1(W)
    p2 = part2(W)
    p3 = part3()
    print()
    print("GATE VERDICT: part1 output bitwise=%s  part2 anti-vacuity=%s  "
          "part3 fallback=%s" % (p1, p2, p3))
    if not (p1 and p2 and p3):
        print("GATE FAILED. No timing number from this patch is admissible until this "
              "passes; a faster kernel that is not correct is not a result.")
        return 1
    print("GATE PASSED on both grids, so timing may now be read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
