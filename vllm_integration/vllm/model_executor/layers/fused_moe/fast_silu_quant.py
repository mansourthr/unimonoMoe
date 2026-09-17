# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad-aware fused SiLU-and-mul + FP8 per-block quantize for the MoE intermediate.

Replaces ``ops.silu_and_mul_per_block_quant`` on the expert-parallel Triton path. It
does the same arithmetic, bitwise, but fixes two measured defects in the C kernel on
DeepSeek-V3-shaped MoE layers (H200, hidden 7168, moe_intermediate 2048, E_local 32):

  bandwidth  The C kernel launches one 128-thread block per ``(row, 128-element
             group)``. Each thread loads exactly one bf16 gate element and one bf16 up
             element, so every global access is 2 bytes wide, and the group max goes
             through a 7-step ``__shared__`` tree with a ``__syncthreads`` per step.
             Measured, it reaches 807-824 GB/s where a device-to-device copy of the same
             byte volume reaches 3755-4033 GB/s: 20-22% of achievable. This kernel gives
             each program one whole row, so the gate and up loads are contiguous and
             full-width and the group max is a register reduction with no barrier.

  padding    The C kernel has no pad awareness, so it processes every slot row of
             ``intermediate_cache1``. Under DeepSeek-V3 routing (n_group 8, topk_group
             4, group == EP rank) a token that reaches a rank contributes 8/4 = 2 local
             experts on average, so roughly 75% of the rows are not owned by this rank.
             Passing ``topk_ids`` and ``expert_map`` skips those rows, using the same
             ownership test as ``_swiglu_limit_pad_aware_kernel`` in
             ``fused_moe/utils.py``, which is the tree's existing precedent for this on
             the clamped SwiGLU path.

Measured on one idle H200 against the C kernel, median of 15, bitwise identical output:

    recv tokens  width   rows    C kernel    this kernel
           4019      8   32152    409.9 us        65.8 us   -84.0%
           4019      4   16076    217.6 us        55.7 us   -74.4%
           8127      8   65016    813.1 us       109.1 us   -86.6%
           8127      4   32508    414.6 us        91.3 us   -78.0%

The unwritten rows are safe, but only because nothing reads them, and that holds for a
specific reason on each grid rather than in general. On the ragged grid
(``ignore_invalid_experts=True``) ``sorted_token_ids`` contains owned slots only, so the
second GEMM never indexes a skipped row. On the full-rectangle grid the skipped rows do
appear in ``sorted_token_ids``, but their ``expert_ids`` entry is -1 and
``fused_moe_kernel`` takes the ``off_experts == -1`` branch (``fused_moe.py:167`` and
``:424``), which calls ``write_zeros_to_output`` and returns without loading A or the A
scales. Rows are independent in this kernel, so a skipped row cannot perturb an owned
one either. Any future caller that reads the full rectangle must pass ``topk_ids=None``.

Bitwise identity required three specific lowerings and was established by measurement
across 18 variants, not by inspection. Triton's ``/`` does not lower to ``div.rn.f32``:
with the default divide, 73562 of 131072 group scales land one ULP away from the C
kernel. Using ``div_rn`` for the scale and the quantize but not for the sigmoid's own
reciprocal still leaves 4506, and ``div_rn`` everywhere with ``tl.exp`` instead of
``libdevice.exp`` still leaves 2288. All three divides as ``div_rn`` plus
``libdevice.exp`` gives 0 of 16777216 elements and 0 of 131072 scales. Changing any of
them back reintroduces a deviation that no cosine check would catch, so they are not
stylistic choices.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul_block_quant_kernel(
    input_ptr,  # [num_rows, 2 * H], gate || up
    out_ptr,  # [num_rows, H], fp8
    scale_ptr,  # [num_rows, H // G], fp32
    topk_ids_ptr,  # [num_rows] flattened, global expert ids, or None
    expert_map_ptr,  # global -> local expert id, -1 if not local, or None
    num_rows,
    H: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    QMAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    MASKED: tl.constexpr,
):
    row = tl.program_id(0)
    if MASKED:
        # Same ownership test as _swiglu_limit_pad_aware_kernel: the slot's global
        # expert id must be non-negative and must map to a local expert.
        eid = tl.load(topk_ids_ptr + row)
        if eid < 0:
            return
        lid = tl.load(expert_map_ptr + eid)
        if lid < 0:
            return

    h = tl.arange(0, H)
    base = row.to(tl.int64) * (2 * H)
    gate = tl.load(input_ptr + base + h).to(tl.float32)
    up = tl.load(input_ptr + base + H + h).to(tl.float32)

    # The C kernel's exact expression: the sigmoid is its own division, then two
    # multiplies. div_rn and libdevice.exp are required for bitwise identity.
    sigmoid_gate = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    result = gate * sigmoid_gate * up

    result = tl.reshape(result, (NG, G))
    group_max = tl.max(tl.abs(result), axis=1)
    group_scale = tl.maximum(tl.math.div_rn(group_max, QMAX), MIN_SCALE)
    q = tl.math.div_rn(result, group_scale[:, None])
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX)

    tl.store(
        out_ptr + row.to(tl.int64) * H + h,
        tl.reshape(q, (H,)).to(out_ptr.dtype.element_ty),
    )
    tl.store(scale_ptr + row.to(tl.int64) * NG + tl.arange(0, NG), group_scale)


def _supported(x: torch.Tensor, group_size: int, quant_dtype: torch.dtype) -> bool:
    """Whether this kernel can serve the call; the caller falls back to the C op.

    The row must fit in registers as fp32, and H must be a power of two so
    ``tl.reshape`` into (groups, group_size) is exact. Both hold for every MoE shape
    this path currently sees; the guard exists so an unusual shape degrades to the C
    kernel instead of failing to compile.
    """
    if x.dim() != 2 or not x.is_contiguous():
        return False
    if quant_dtype != torch.float8_e4m3fn:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.shape[1] % 2 != 0:
        return False
    h = x.shape[1] // 2
    # A power-of-two H no wider than the compiled bound, holding a whole number
    # of quantization groups: the kernel indexes the group dimension by shift.
    return 0 < h <= 8192 and h % group_size == 0 and (h & (h - 1)) == 0


def silu_and_mul_per_block_quant_pad_aware(
    x: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU-and-mul and FP8 per-block quantize, skipping unowned rows.

    ``x`` is ``[num_rows, 2 * H]`` with gate in the first half. Returns
    ``(quantized [num_rows, H], scales [num_rows, H // group_size])`` in the
    same layout as ``ops.silu_and_mul_per_block_quant`` with
    ``is_scale_transposed=False``.

    When ``topk_ids`` is given (and ``expert_map`` with it) the rows whose expert is not
    local are left UNWRITTEN rather than zeroed, which is what makes the saving real.
    See the module docstring for why the callers in this tree never read them.
    """
    from vllm import _custom_ops as ops

    if not _supported(x, group_size, quant_dtype):
        return ops.silu_and_mul_per_block_quant(
            x, group_size=group_size, quant_dtype=quant_dtype
        )

    num_rows, gate_up = x.shape
    h = gate_up // 2
    ngroup = h // group_size
    out = torch.empty(num_rows, h, device=x.device, dtype=quant_dtype)
    scales = torch.empty(num_rows, ngroup, device=x.device, dtype=torch.float32)
    if num_rows == 0:
        return out, scales

    masked = topk_ids is not None and expert_map is not None
    if masked:
        # A mismatch here would silently mask the wrong rows, so it is checked rather
        # than assumed: topk_ids is [num_tokens, top_k] and x is one row per slot.
        assert topk_ids.numel() == num_rows, (
            f"topk_ids has {topk_ids.numel()} slots but the intermediate has "
            f"{num_rows} rows"
        )
        tids = topk_ids.reshape(-1)
    else:
        tids = None

    qmax = torch.finfo(quant_dtype).max
    _silu_mul_block_quant_kernel[(num_rows,)](
        x,
        out,
        scales,
        tids,
        expert_map if masked else None,
        num_rows,
        H=h,
        G=group_size,
        NG=ngroup,
        QMAX=qmax,
        # matches min_scaling_factor<T>::val() at csrc/quantization/utils.cuh:50-53
        MIN_SCALE=1.0 / (qmax * 512.0),
        MASKED=masked,
        num_warps=4,
    )
    return out, scales
