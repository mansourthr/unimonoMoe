# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe._ep_timer import begin as _ep_begin
from vllm.model_executor.layers.fused_moe._ep_timer import end as _ep_end
from vllm.model_executor.layers.fused_moe._ep_timer import span as _ep_span
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    _USE_LAYERNAME,
    LayerName,
    direct_register_custom_op,
)

logger = init_logger(__name__)

# Diagnostic budget for the shared-expert fold probe, see _fold_debug_probe.
# Off unless the environment variable is set, and spent on the first fused call
# so that a debug run costs one extra kernel call and two extra collectives.
_FOLD_DEBUG_LEFT = int(os.environ.get("VLLM_PREFILL_MONOKERNEL_TP_FOLD_DEBUG") or 0)


def register_layer_for_moe_forward_op(
    vllm_config: VllmConfig,
    layer: "MoERunner",
):
    # For smuggling this layer into the fused moe custom op
    prefix = layer.layer_name
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = layer
    compilation_config.static_all_moe_layers.append(prefix)


def get_layer_from_name(layer_name: str) -> MoERunnerInterface:
    forward_context: ForwardContext = get_forward_context()
    if not _USE_LAYERNAME and layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{vllm.moe_forward, vllm.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1
    layer = forward_context.no_compile_layers[layer_name]
    assert isinstance(layer, MoERunnerInterface)
    return layer


# On torch >= 2.11, layer_name is a hoisted LayerName opaque object;
# on older versions it remains a plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | LayerName
else:
    _layer_name_type = LayerName if _USE_LAYERNAME else str


@torch.compiler.assume_constant_result
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(layer_name, LayerName):
        return layer_name.value
    elif isinstance(layer_name, FakeScriptObject):
        return layer_name.real_obj.value
    return layer_name


# Note: _moe_forward and _moe_forward_shared should not contain any
# implementation details, They should merely pass along control to
# the runner's '_forward_impl' method.
# These functions should never be called directly since they do not
# include all the functionality of the MoE layer.
def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> torch.Tensor:
    # `hidden_dim_unpadded > 0` only on the TRT-LLM MXFP4 path, where the
    # real kernel writes narrower than `hidden_states.shape[-1]`. Plumbed
    # as an op arg (not peeked from the layer registry) to keep the fake
    # a pure shape function of its inputs and preserve subgraph dedup.
    if hidden_dim_unpadded > 0:
        return hidden_states.new_empty((*hidden_states.shape[:-1], hidden_dim_unpadded))
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    return layer._forward_impl(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `fused_out`: see `_moe_forward_fake` for hidden_dim_unpadded semantics.
    # `shared_out`: matches `shared_experts_input` if provided (latent MoE),
    # else `hidden_states`.
    if hidden_dim_unpadded > 0:
        fused_out = hidden_states.new_empty(
            (*hidden_states.shape[:-1], hidden_dim_unpadded)
        )
    else:
        fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


# NOTE: `moe_forward` and `moe_forward_shared` being opaque custom ops is a
# load-bearing assumption for the MoE-LoRA dual-stream path.
direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def _unpack(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(result, tuple):
        return result
    else:
        return (None, result)


# --- env gated routing probe (off unless VLLM_EP_ROUTE_DUMP is set) ---------
_EP_ROUTE_DUMP = os.environ.get("VLLM_EP_ROUTE_DUMP", "")
# Read once into a module constant, so that when timing is off Dynamo folds the
# `if _EP_TIME_ON:` guards below to false and the probe calls leave no trace in the
# traced graph at all.
_EP_TIME_ON = bool(os.environ.get("VLLM_EP_TIME_DUMP", ""))
_ep_route_state: dict = {}


def _ep_route_record(layer_name, topk_ids):
    """Accumulate per layer expert and destination rank statistics.

    Placement agnostic: the probe runs after the EPLB remap, so `topk_ids` are
    PHYSICAL slot ids. `VLLM_EP_ROUTE_EPHYS` gives the physical slot count and the
    per-rank stride follows from it, which makes `rows_lin` the real per-rank rows
    and `d_lin` the real destination rank fan-out under any placement. Nothing here
    is a timing measurement.
    """
    import torch as _t

    ids = topk_ids.detach().to(_t.int64)
    if ids.dim() != 2:
        return
    T, K = ids.shape
    ep = 8
    Elog = int(_ep_route_state.get("Elog", 0)) or 256
    _ep_route_state["Elog"] = Elog
    Ephys = int(_ep_route_state.get("Ephys", 0))
    if Ephys == 0:
        Ephys = int(os.environ.get("VLLM_EP_ROUTE_EPHYS", str(Elog)))
        _ep_route_state["Ephys"] = Ephys
    per = Ephys // ep
    glog = Elog // ep

    hist = _t.bincount(ids.reshape(-1), minlength=Ephys)
    # actual owner rank of each chosen physical slot
    r_lin = ids // per
    # the scattered reference placement, on the LOGICAL id so it stays comparable
    r_rr = (ids % Elog) % ep
    oh_lin = _t.zeros(T, ep, device=ids.device, dtype=_t.int64)
    oh_lin.scatter_(1, r_lin, 1)
    oh_rr = _t.zeros(T, ep, device=ids.device, dtype=_t.int64)
    oh_rr.scatter_(1, r_rr, 1)
    # exact per token LOGICAL expert-group set as an 8 bit mask, 256 bins total
    g = (ids % Elog) // glog
    oh_g = _t.zeros(T, ep, device=ids.device, dtype=_t.int64)
    oh_g.scatter_(1, g, 1)
    _w = 1 << _t.arange(ep, device=ids.device, dtype=_t.int64)
    gmask = _t.bincount((oh_g * _w).sum(1), minlength=1 << ep)
    d_lin = _t.bincount(oh_lin.sum(1), minlength=ep + 1)
    d_rr = _t.bincount(oh_rr.sum(1), minlength=ep + 1)
    rows_lin = _t.bincount(r_lin.reshape(-1), minlength=ep)
    rows_rr = _t.bincount(r_rr.reshape(-1), minlength=ep)

    # one record per call: the engine runs dummy profile passes before the
    # real prefill and pooling them together corrupts the routing statistics
    seen = _ep_route_state.setdefault("_n", {})
    lname = str(layer_name)
    seen[lname] = seen.get(lname, 0) + 1
    key = f"{lname}|T{T}|c{seen[lname]}"
    rec = _ep_route_state.setdefault(
        key,
        {
            "calls": 0,
            "T": T,
            "K": K,
            "Ephys": Ephys,
            "idmax": 0,
            "hist": [0] * Ephys,
            "d_lin": [0] * (ep + 1),
            "d_rr": [0] * (ep + 1),
            "rows_lin": [0] * ep,
            "rows_rr": [0] * ep,
            "gmask": [0] * (1 << ep),
        },
    )
    rec["calls"] += 1
    rec["idmax"] = max(rec["idmax"], int(ids.reshape(-1).max().item()))
    for name, t in (
        ("hist", hist),
        ("d_lin", d_lin),
        ("d_rr", d_rr),
        ("rows_lin", rows_lin),
        ("rows_rr", rows_rr),
        ("gmask", gmask),
    ):
        cur = rec[name]
        vals = t.tolist()
        for i, v in enumerate(vals):
            cur[i] += int(v)


def _ep_route_flush():
    if not _EP_ROUTE_DUMP or not _ep_route_state:
        return
    import json as _j

    # The TP rank is 0 on every worker under TP=1/PCP=8, so all eight would
    # write one file. Use the global rank, which is distinct by construction.
    rank = -1
    try:
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            rank = _dist.get_rank()
    except Exception:
        rank = -1
    if rank < 0:
        rank = int(os.environ.get("VLLM_DP_RANK", os.environ.get("RANK", "0")))
    out = {k: v for k, v in _ep_route_state.items() if isinstance(v, dict) and "T" in v}
    with open(f"{_EP_ROUTE_DUMP}.rank{rank}.json", "w") as f:
        _j.dump({"E": _ep_route_state.get("E", 256), "layers": out}, f)


if _EP_ROUTE_DUMP:
    import atexit as _atexit

    _atexit.register(_ep_route_flush)
# --- end routing probe -----------------------------------------------------


class MoERunner(MoERunnerInterface):
    """
    Standard MoE runner implementation for executing Mixture of Experts layers.

    This is the primary concrete implementation of MoE execution logic, providing
    comprehensive support for standard MoE operations. It handles:
    - Expert routing and token dispatching using various routing strategies
    - Shared experts computation with optional parallel execution using CUDA streams
    - Tensor model parallel and expert parallel operations
    - Multiple quantization methods and optimized kernel selection
    - Both monolithic and decomposed expert execution paths
    - Integration with various parallel execution modes (TP, EP, DP)

    The runner orchestrates the complete MoE forward pass including routing tokens
    to experts, executing expert computations in parallel, and combining results.
    It supports advanced features like overlapped execution of shared experts,
    optimized kernels for different parallel configurations, and seamless
    integration with vLLM's distributed execution framework.

    Eventually, this class may be split into more specialized implementations
    for different configurations (e.g., with/without shared experts, gates, etc.).
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: torch.nn.Module | None = None,
        shared_experts: torch.nn.Module | None = None,
        shared_expert_gate: torch.nn.Module | None = None,
        routed_input_transform: torch.nn.Module | None = None,
        routed_output_transform: torch.nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self.gate = gate
        self.shared_expert_gate = shared_expert_gate
        self.routed_experts = routed_experts
        self.enable_dbo = enable_dbo

        # When both gates are present and FSE is enabled, fuse their
        # weight matrices into [num_experts + num_shared, hidden] so one
        # F.linear produces combined logits. The topk kernel can then
        # apply routing softmax and shared expert activation (sigmoid)
        # in a single launch.
        self._fse_fuse_gate = gate is not None and shared_expert_gate is not None
        self._combined_gate_weight: torch.Tensor | None = None

        self._shared_experts: SharedExperts | None = None
        if shared_experts is not None:
            can_overlap = lambda: self._quant_method.mk_can_overlap_shared_experts
            self._shared_experts = SharedExperts(
                shared_experts,
                moe_config=moe_config,
                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )

        # Needed for string -> MoERunner layer lookup in custom ops.
        self.layer_name = layer_name

        self._forward_entry = self._select_forward()

        # For smuggling this layer into the fused moe custom op
        register_layer_for_moe_forward_op(get_current_vllm_config(), self)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        return self.routed_experts.load_weights(weights)

    def _select_forward(self) -> Callable:
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: Once the OOM issue for the TPU backend is resolved, we
            # will switch to using the moe_forward custom op.
            # Note: CPU doesn't require wrapped _forward_impl.
            return _moe_forward if self._shared_experts is None else _moe_forward_shared

        return (
            torch.ops.vllm.moe_forward
            if self._shared_experts is None
            else torch.ops.vllm.moe_forward_shared
        )

    @property
    def shared_experts(self) -> SharedExperts | None:
        return self._shared_experts

    # TODO(bnell): Temporary hack. Get rid of this.
    def _replace_quant_method(self, quant_method: FusedMoEMethodBase):
        self.routed_experts._replace_quant_method(quant_method)

    # TODO(bnell): Hack for elastic_ep. Get rid of this
    def _set_moe_config(self, new_moe_config: FusedMoEConfig):
        self.moe_config = new_moe_config
        self.routed_experts._set_moe_config(new_moe_config)
        if self._shared_experts is not None:
            self._shared_experts._set_moe_config(new_moe_config)

    def _maybe_fuse_gate_weights(self):
        """Fuse router and shared expert gate weights on first call.

        Cannot be done at __init__ because gate weights are loaded after
        module construction (via weight_loader). Called once from
        _forward_impl before the first forward pass.
        """
        if self._combined_gate_weight is None:
            assert self.gate is not None and self.shared_expert_gate is not None
            self._combined_gate_weight = torch.cat(
                [self.gate.weight, self.shared_expert_gate.weight],
                dim=0,
            )

    @property
    def _quant_method(self) -> FusedMoEMethodBase:
        return self.routed_experts.quant_method

    def apply_routed_input_transform(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply transform for routed experts (e.g., latent projection).

        This is called by MoERunner.forward_native. The original hidden_states
        is saved separately so shared experts get [S, hidden_size] while
        routed experts get the transformed [S, moe_latent_size].

        Returns (possibly transformed) hidden states and the input for shared
        experts (or None if there are no shared experts).
        """
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear returns (output, extra_bias) tuple.
            # We only need the output tensor; extra_bias is not used here.
            if isinstance(result, tuple):
                return result[0], hidden_states
            return result, hidden_states

        return (
            hidden_states,
            hidden_states if self._shared_experts is not None else None,
        )

    def apply_routed_output_transform(
        self,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transform to routed expert output (e.g., latent to full dim).

        Used by latent MoE models (e.g., NemotronH) where routed experts
        operate in a compressed latent space and need projection back to
        the full hidden dimension before combining with shared expert output.
        """
        if self.routed_output_transform is not None:
            r = self.routed_output_transform(fused_output)
            fused_output = r[0] if isinstance(r, tuple) else r
        return fused_output

    def _maybe_apply_routed_scale_to_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Apply routed_scaling_factor to the output with FP16 overflow
        protection.

        Scale the fused expert output by routed_scaling_factor. For FP16,
        avoid overflow by dividing shared_output by the scale instead
        (the decoder layer compensates with matching divisions).
        """
        if self.routed_scaling_factor != 1.0:
            if fused_output.dtype != torch.float16 or shared_output is None:
                fused_output *= self.routed_scaling_factor
            elif shared_output is not None:
                shared_output *= 1.0 / self.routed_scaling_factor
        return shared_output, fused_output

    @property
    def _fused_output_is_reduced(self) -> bool:
        return (
            self._quant_method.moe_kernel is not None
            and self._quant_method.moe_kernel.output_is_reduced()
        )

    @property
    def _monokernel_reduce_folded(self) -> bool:
        """Does the monokernel path reduce its own output, at every token count?

        Read from routed_experts.quant_method, the same object
        _maybe_moe_monokernel offers the work to, and through getattr so a quant
        method without a monokernel path is untouched.
        """
        probe = getattr(
            self.routed_experts.quant_method, "monokernel_output_is_reduced", None
        )
        return bool(probe()) if probe is not None else False

    def _fold_debug_probe(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        residual: torch.Tensor,
        fused_out: torch.Tensor,
    ) -> None:
        """Project the folded output onto its two halves and print the fit.

        Diagnostic only, off unless VLLM_PREFILL_MONOKERNEL_TP_FOLD_DEBUG is
        set, and budgeted to the first call per process so it cannot dominate a
        run. See the module patch note for how a and b are read.
        """
        global _FOLD_DEBUG_LEFT
        if _FOLD_DEBUG_LEFT <= 0:
            return
        _FOLD_DEBUG_LEFT -= 1

        import torch.distributed as dist

        # The unfused routed partial, from the same inputs. The fused path
        # declines a residual-free call, so this lands on the ordinary entry.
        routed_partial = self._maybe_moe_monokernel(hidden_states, router_logits)
        if routed_partial is None:
            logger.info("[folddbg] no unfused routed output, probe skipped")
            return

        scale = float(self.routed_scaling_factor)
        r = tensor_model_parallel_all_reduce(routed_partial.float() * scale)
        s = tensor_model_parallel_all_reduce(residual.float())
        got = fused_out.float()
        ref = r + s

        def dot(a, b):
            return float((a * b).sum().item())

        rr, ss, rs = dot(r, r), dot(s, s), dot(r, s)
        gr, gs = dot(got, r), dot(got, s)
        det = rr * ss - rs * rs
        if abs(det) > 0.0:
            a = (gr * ss - gs * rs) / det
            b = (gs * rr - gr * rs) / det
        else:
            a = b = float("nan")

        def cos(x, y):
            n = (x.norm() * y.norm()).item()
            return dot(x, y) / n if n else float("nan")

        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(
            "[folddbg] rank=%d M=%d scale=%.3f fit a=%.6f b=%.6f | "
            "cos(got,ref)=%.9f cos(got,routed)=%.9f cos(got,shared)=%.9f | "
            "norm got=%.4e ref=%.4e routed=%.4e shared=%.4e | "
            "maxabs(got-ref)=%.4e",
            rank,
            hidden_states.shape[0],
            scale,
            a,
            b,
            cos(got, ref),
            cos(got, r),
            cos(got, s),
            got.norm().item(),
            ref.norm().item(),
            r.norm().item(),
            s.norm().item(),
            (got - ref).abs().max().item(),
        )

    @property
    def _monokernel_folds_shared(self) -> bool:
        """Does the monokernel path also carry the SHARED half of the output?

        This is the contract that removes the extra collective, and it is worth
        stating in full because every clause of it is load bearing. When it
        holds, the op hands back ONE complete MoE output in the routed slot:
        already summed across the TP group, with routed_scaling_factor already
        applied, and with the shared half already added in. So the traced side
        must not reduce a shared output, must not apply the scale, and must not
        add the two halves. Doing any of them would be a wrong answer, not a
        slow one.

        WHY IT EXISTS. Folding only the ROUTED reduce into the kernel does not
        remove a collective from this model, it adds one:
        _maybe_reduce_shared_expert_output then reduces the shared half alone,
        at identical shape and identical bytes to the single reduce the baseline
        did of the sum. MEASURED on DS3 TP=8, that was 5.53% slower at 8192 and
        3.54% at 16384.

        SIZE-INDEPENDENT BY CONSTRUCTION, for the same reason as
        _monokernel_reduce_folded: forward() reads it where the token count is
        symbolic and torch.compile bakes the branch from the first size traced.
        The per-size part of the decision lives inside the op, where the real M
        is known, and every branch there ends with exactly one reduction of the
        sum.
        """
        if self._shared_experts is None:
            return False
        if not self._monokernel_reduce_folded:
            return False
        probe = getattr(
            self.routed_experts.quant_method, "monokernel_folds_shared_output", None
        )
        if probe is None or not probe():
            return False
        # Everything the traced side would otherwise put BETWEEN the two halves.
        # A latent output transform sits between the routed output and the add,
        # sequence parallelism reduces elsewhere entirely, and fp16 takes the
        # inverted branch in _maybe_apply_routed_scale_to_output where the scale
        # lands on the shared half instead. None of the three can be expressed
        # as one scale inside the kernel, so the fold declines them.
        return (
            self.routed_output_transform is None
            and not self.moe_config.is_sequence_parallel
            and self.moe_config.in_dtype != torch.float16
        )

    def _maybe_reduce_shared_expert_output(
        self,
        shared_output: torch.Tensor | None,
        fused_output_is_reduced: bool | None = None,
    ) -> torch.Tensor | None:
        """All-reduce shared expert output when the combine kernel already
        reduced fused output.

        * If the combine kernel does the reduction for fused_output, reduce
          shared_output separately. O.w, reduce fused_output+shared_output later.
        * If we have SP (TP=N, DP=M, EP), there is a separate AG step handled
          in the model.
        """
        if fused_output_is_reduced is None:
            fused_output_is_reduced = self._fused_output_is_reduced

        if (
            shared_output is not None
            and not self.moe_config.is_sequence_parallel
            and fused_output_is_reduced
        ):
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        return shared_output

    def _maybe_reduce_routed_output_before_transform(
        self,
        fused_output: torch.Tensor,
        fused_output_is_reduced: bool,
    ) -> tuple[torch.Tensor, bool]:
        """All-reduce latent routed output before its output transform.

        Latent MoE output transforms may contain non-linear ops, e.g. RMSNorm.
        TP partial routed outputs must be summed in latent space before such
        transforms are applied.
        """
        if (
            self.routed_output_transform is not None
            and not self.moe_config.is_sequence_parallel
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not fused_output_is_reduced
        ):
            fused_output = tensor_model_parallel_all_reduce(fused_output)
            fused_output_is_reduced = True
        return fused_output, fused_output_is_reduced

    def _maybe_reduce_final_output(
        self,
        states: torch.Tensor,
        trunc_size: int | None,
        output_is_reduced: bool | None = None,
    ) -> torch.Tensor:
        """All-reduce the combined output if needed.

        This is the "late" all-reduce path. When neither fused nor shared
        output was individually reduced, the combined sum is all-reduced
        here. Skipped when sequence-parallel is active (SP handles its
        own reduction) or when the early path already reduced both outputs.
        """
        # skip_final_all_reduce must not coexist with a pre-reduced fused
        # output. This should be enforced by MoE config initialization.
        if self.moe_config.skip_final_all_reduce:
            assert not self._fused_output_is_reduced, (
                "skip_final_all_reduce requires an un-reduced fused output"
            )

        # We don't need to reduce the final output if:
        # - We are not running with TP or DP
        # - The MK already reduced the fused output itself.
        if output_is_reduced is None:
            output_is_reduced = self._fused_output_is_reduced

        if (
            not self.moe_config.is_sequence_parallel
            and not self.moe_config.skip_final_all_reduce
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
            and not output_is_reduced
        ):
            with _ep_span("reduce", "moe"):
                states = tensor_model_parallel_all_reduce(states)

        return states[..., :trunc_size] if trunc_size is not None else states

    def _encode_layer_name(self) -> str | LayerName:
        if _USE_LAYERNAME:
            return LayerName(self.layer_name)
        # Can be unavailable or None in unittests
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def _maybe_pad_hidden_states(
        self,
        shared_experts_input: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, int | None, int | None]:
        """Pad hidden_states to moe_config.hidden_dim and compute the
        original dimension for later truncation.

        For latent MoE, the routed hidden_states may be smaller than
        hidden_dim. Padding ensures uniform tensor sizes through the
        fused MoE kernel. The returned trunc_size is used by
        _maybe_reduce_final_output to strip the padding from the result.
        """
        shared_experts_hidden_dim = (
            shared_experts_input.shape[-1] if shared_experts_input is not None else 0
        )
        transformed_hidden_dim: int | None = hidden_states.shape[-1]
        if (
            not self._quant_method.skip_forward_padding
            and self.moe_config.hidden_dim != transformed_hidden_dim
        ):
            assert transformed_hidden_dim is not None
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # Truncation sizes for stripping kernel padding from the output.
        # None means no truncation needed (no padding was applied).
        #
        # Two truncation points exist in forward():
        #   pre_xform:  applied to fused_output BEFORE routed_output_transform
        #   post_xform: applied to the final result AFTER all-reduce
        #
        # MoE with routed output transform or shared experts:
        #   - pre_xform applies if the transform needs unpadded routed output
        #     or shared+routed add needs matching hidden dims. For Nemotron-3
        #     Nano, TRTLLM NVFP4 pads routed MoE hidden dim 2688->2816, while
        #     shared output stays 2688.
        #   - post_xform uses shared_experts_hidden_dim when transform and shared
        #     experts make the final output full hidden dim.
        #
        # Standard MoE / MoE without transforms (GPT-OSS, Mixtral):
        #   - pre_xform is None (no early truncation)
        #   - post_xform strips padding after all-reduce (or None if unpadded)
        if transformed_hidden_dim == hidden_states.shape[-1]:
            transformed_hidden_dim = None

        pre_xform_trunc_size = None
        if self.routed_output_transform is not None or shared_experts_hidden_dim > 0:
            pre_xform_trunc_size = transformed_hidden_dim
        post_xform_trunc_size = transformed_hidden_dim
        if self.routed_output_transform is not None and shared_experts_hidden_dim > 0:
            post_xform_trunc_size = shared_experts_hidden_dim

        return hidden_states, pre_xform_trunc_size, post_xform_trunc_size

    def _maybe_apply_shared_experts(
        self,
        shared_experts_input: torch.Tensor | None,
        order: SharedExpertsOrder,
    ):
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts(shared_experts_input, order)

    def _maybe_moe_monokernel(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_scale: float = 1.0,
    ) -> torch.Tensor | None:
        """Offer the work to the standalone prefill MoE monokernel.

        The kernel selects experts itself from the raw router logits, so it has
        to be offered those logits before select_experts runs and cannot be
        reached from the modular apply path, which only ever sees weights and
        ids that routing has already produced. Returning None means "not
        eligible" and leaves the ordinary paths below untouched.

        The quant method owns the eligibility decision, since that depends on
        weight geometry and on the backend chosen at load time. The fast path is
        declined when the modular kernel would run the shared experts inside
        itself, because bypassing it would then silently drop that work.
        """
        quant_method = self.routed_experts.quant_method
        try_monokernel = getattr(quant_method, "try_moe_monokernel", None)
        if try_monokernel is None or quant_method.mk_can_overlap_shared_experts:
            return None
        if residual is None:
            # The unchanged call, so a quant method that predates the fold and
            # takes three arguments is untouched. The extra arguments are only
            # passed when the fold asked for them, which requires the quant
            # method to have answered monokernel_folds_shared_output.
            return try_monokernel(self.routed_experts, hidden_states, router_logits)
        return try_monokernel(
            self.routed_experts,
            hidden_states,
            router_logits,
            residual=residual,
            out_scale=out_scale,
        )

    def _apply_quant_method(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run expert routing and the fused MoE kernel via the quant method.

        Orchestrates shared expert execution (before/after), expert selection
        via the router, and the actual fused MoE computation. Returns
        (shared_expert_output, fused_expert_output).
        """
        self._maybe_apply_shared_experts(
            shared_experts_input, SharedExpertsOrder.NO_OVERLAP
        )

        # The shared half, offered to the routed kernel so its own collective
        # carries the complete MoE output. PEEK rather than pop: the ordinary
        # consumer below still pops it, so nothing about the op's return
        # contract changes. At the token counts the fused epilogue accepts, the
        # shared expert has already run to completion on the main stream in the
        # call just above, because VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        # sends anything wider than 256 tokens down the NO_OVERLAP order. Below
        # that it runs after the routed kernel and there is nothing to fold
        # yet, which is a None here and a combine further down.
        fold_shared = self._monokernel_folds_shared
        residual: torch.Tensor | None = None
        out_scale = 1.0
        if fold_shared:
            assert self._shared_experts is not None
            residual = self._shared_experts.pending_output
            if residual is not None:
                out_scale = self.routed_scaling_factor

        # Prefill monokernel first, since it routes internally and so has to
        # see the logits before select_experts consumes them. None means the
        # kernel declined and the original two branches run unchanged.
        fused_out = self._maybe_moe_monokernel(
            hidden_states, router_logits, residual, out_scale
        )

        if fused_out is not None:
            pass
        elif self.routed_experts.quant_method.is_monolithic:
            # Monolithic kernels: pass router_logits to routed_experts
            fused_out = self.routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )
        else:
            # Modular kernels: select experts first, then call routed_experts
            with _ep_span("route", self.layer_name):
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_indices_dtype=self._quant_method.topk_indices_dtype,
                    input_ids=input_ids,
                )

            if _EP_ROUTE_DUMP:
                _ep_route_record(self.layer_name, topk_ids)

            with _ep_span("layer", self.layer_name):
                fused_out = self.routed_experts.forward_modular(
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=self._shared_experts,
                    shared_experts_input=shared_experts_input,
                )

        self._maybe_apply_shared_experts(
            shared_experts_input,
            SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
        )

        # Reduce the routed output here when the monokernel path declares itself
        # reduced but this particular call did not go through the fused epilogue.
        # forward() reads that declaration on the traced side with a symbolic
        # token count, so it cannot vary with M; the sizes the epilogue declines
        # get their reduce here instead, and _maybe_reduce_final_output skips
        # one in exchange. Placed after the shared experts are launched so the
        # collective overlaps their side stream rather than delaying it.
        shared_out = (
            self._shared_experts.output if self._shared_experts is not None else None
        )

        if _FOLD_DEBUG_LEFT > 0 and fold_shared and residual is not None:
            from vllm.model_executor.layers.fused_moe.monokernel_tp import (
                output_is_fused as _dbg_is_fused,
            )

            if _dbg_is_fused(fused_out):
                self._fold_debug_probe(
                    hidden_states, router_logits, residual, fused_out
                )

        if self._monokernel_reduce_folded and not self._fused_output_is_reduced:
            from vllm.model_executor.layers.fused_moe.monokernel_tp import (
                output_is_fused,
            )

            # output_is_fused is the ground truth for whether THIS call went
            # through the fused epilogue, read from the destination's address
            # rather than from a flag, so it is correct inside a replayed graph.
            # The fused epilogue is only entered with a residual in hand
            # (fused_tp_launch refuses otherwise), so a fused output is a folded
            # output and the two questions have one answer.
            was_fused = output_is_fused(fused_out)

            if fold_shared and not was_fused and shared_out is not None:
                # The kernel took the ordinary path at this token count, so the
                # two halves are both still partial. Combining them HERE and
                # reducing the sum once below is the baseline's own shape: one
                # collective, not two. The scale goes on the routed half only,
                # which is what _maybe_apply_routed_scale_to_output does and
                # what the traced side is skipping in exchange.
                if self.routed_scaling_factor != 1.0:
                    fused_out = fused_out * self.routed_scaling_factor
                fused_out = fused_out + shared_out
            elif fold_shared and was_fused and residual is None:
                # Structurally unreachable: the fused sizes are far above the
                # aux-stream threshold, and fused_tp_launch declines without a
                # residual. Raise rather than add an unreduced shared half to a
                # reduced routed one, which would be a silent wrong answer.
                raise RuntimeError(
                    "the fused MoE epilogue ran without the shared half folded "
                    "in; the shared output was not available at launch"
                )

            if not was_fused:
                fused_out = tensor_model_parallel_all_reduce(fused_out)

        return (
            shared_out,
            fused_out,
        )

    def _sequence_parallel_context(self):
        """Return a context manager for sequence-parallel token
        redistribution.

        When sequence parallelism is active, returns a context that handles
        local size tracking for proper token scatter/gather. Otherwise
        returns a no-op context.
        """
        ctx = get_forward_context()
        return (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

    def _maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor | None,
    ):
        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the RoutedExperts via
        #        separate cuda stream)
        if self._shared_experts is not None:
            assert shared_experts_input is not None
            self._shared_experts.maybe_sync_shared_experts_stream(shared_experts_input)

    def _maybe_add_zero_expert_output(
        self,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """Add the zero expert's contribution to the final result.

        When a ZeroExpertRouter is used, it computes a bias-like output
        from the "zero expert" that is added to the combined routed+shared
        expert output.
        """
        if isinstance(self.router, ZeroExpertRouter):
            zero_expert_output = self.router.zero_expert_output
            assert zero_expert_output is not None
            result = result + zero_expert_output
        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the fused moe layer.

        Input:
        - hidden_states
        - router_logits

        Output:
        - The new hidden_states.

        Calling sequence
        - forward
          - self._forward_entry (_moe_forward or _moe_forward_shared custom op)
            - _forward_impl

        Note: The existence of _moe_forward and _moe_forward_shared custom ops are due
        to the following reason:
        1. pytorch cannot handle union types in custom op signatures so
           _moe_forward and _moe_forward_shared must be split.
        """

        # Apply transform for routed experts (e.g., latent projection for
        # latent MoE). When the caller pre-applies the routed input transform
        # outside the runner (e.g. to overlap it on a separate stream), it
        # passes the already-transformed routed input as ``hidden_states`` and
        # the original hidden states as ``shared_experts_input``; skip the
        # transform in that case so shared experts still see the original input.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        # Record before `_maybe_pad_hidden_states` pads activations to match
        # `moe_config.hidden_dim`, e.g. after `align_trtllm_fp4_moe_hidden_dim_for_fi`
        # so routed output can be trimmed before
        # shared+routed add / latent up proj if needed.

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        # rmoe: the one span that means the same thing in all four campaign arms,
        # namely router logits in hand through to the complete reduced MoE output. It
        # has to start here, outside _forward_entry, because the monokernel arm does
        # its routing, its expert work and its output collective inside that single
        # call.
        if _EP_TIME_ON:
            _ep_begin("rmoe", self.layer_name)
        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        #
        # Note: there are two all-reduce points below. They are mutually
        # exclusive, controlled by _fused_output_is_reduced
        #  - When True: the combine kernel already reduced fused_output,
        #    so we reduce shared_output here to match, then skip the
        #    all-reduce in _maybe_reduce_final_output.
        #  - When False: neither output is reduced yet, so we combine
        #    them first and all-reduce the sum in _maybe_reduce_final_output.

        # Extract outputs from result
        shared_output, fused_output = _unpack(result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        if self._monokernel_folds_shared:
            # The op folded the shared half into the routed kernel's epilogue
            # and reduced the sum once, so what came back in the routed slot is
            # already the complete MoE output. The shared slot still holds the
            # tensor the op computed, because the op's fake declares two tensors
            # and the compiled graph is built from that declaration; it is
            # DROPPED here rather than reduced, scaled and added a second time.
            if og_hidden_dim_pre_xform is not None:
                raise RuntimeError(
                    "the shared-expert fold cannot be combined with a padded "
                    "routed hidden dim: the kernel added the two halves at full "
                    "width before the truncation point"
                )
            shared_output = None

        # _monokernel_reduce_folded is the monokernel path's own claim, and it
        # is deliberately not a function of the token count: see
        # Fp8MoEMethod.monokernel_output_is_reduced. Whichever branch ran inside
        # the op, the routed output is reduced by the time it gets here.
        fused_output_is_reduced = (
            self._fused_output_is_reduced or self._monokernel_reduce_folded
        )

        # Latent routed output has to be reduced before output transform,
        # because the transform may include non-linear normalization.
        fused_output, fused_output_is_reduced = (
            self._maybe_reduce_routed_output_before_transform(
                fused_output,
                fused_output_is_reduced,
            )
        )

        # If routed output is already reduced, reduce shared to match.
        # See note above re: the two all-reduce points.
        shared_output = self._maybe_reduce_shared_expert_output(
            shared_output, fused_output_is_reduced
        )

        if not self._monokernel_folds_shared:
            # Under the fold the scale was already applied, in fp32 inside the
            # kernel's Phase-3 store and before the round to bf16. Applying it
            # again here would double the routed half.
            shared_output, fused_output = self._maybe_apply_routed_scale_to_output(
                shared_output, fused_output
            )

        # Apply output transform (e.g. latent -> full dim)
        fused_output = self.apply_routed_output_transform(fused_output)

        if shared_output is not None:
            result = shared_output + fused_output
        else:
            result = fused_output

        result = self._maybe_reduce_final_output(
            result, og_hidden_dim_post_xform, fused_output_is_reduced
        )
        # Closed after the reduce, so the arms that pay an external all-reduce are
        # charged for it and the arm that folds it into Phase 3 shows the saving
        # instead of hiding it.
        if _EP_TIME_ON:
            _ep_end()

        return self._maybe_add_zero_expert_output(result)

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return (
            self.moe_config.dp_size > 1 or self.moe_config.is_sequence_parallel
        ) and not self._quant_method.supports_internal_mk

    def _maybe_dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # For naive dispatch/combine Dp/Ep, dispatch the hidden states and
        # router logits to all experts.
        # NOTE: this will be removed once all kernels are migrated into the
        # MoEKernel framework.
        if self.do_naive_dispatch_combine:
            result = get_ep_group().dispatch_router_logits(
                hidden_states,
                router_logits,
                self.moe_config.is_sequence_parallel,
            )
            assert len(result) == 2
            hidden_states, router_logits = result

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            hidden_states = get_pcp_group().all_gather(hidden_states, dim=0)
            router_logits = get_pcp_group().all_gather(router_logits, dim=0)

        return hidden_states, router_logits

    def _maybe_combine(
        self,
        shared_output: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor]:
        if self.do_naive_dispatch_combine:
            hidden_states = get_ep_group().combine(
                hidden_states, self.moe_config.is_sequence_parallel
            )

        if (
            self.moe_config.pcp_size > 1
            and not self.moe_config.moe_parallel_config.use_all2all_kernels
        ):
            hidden_states = get_pcp_group().reduce_scatter(hidden_states, dim=0)

        if self.shared_experts is not None:
            assert shared_output is not None
            return shared_output, hidden_states
        else:
            return hidden_states

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Entry point called by the custom op to run the MoE computation.

        Handles pre-dispatch setup (gate application, external shared expert
        triggering, quant config init) then performs the following steps
        within the sequence-parallel context.

        - Performs expert routing
        - fused MoE kernel execution
        - shared expert computation.

        Returns a single tensor of combined fused and shared output (if present).
        """
        # TODO(bnell): this can be removed after MK migration is complete.
        self.routed_experts._ensure_moe_quant_config_init()

        # Sync aux and main stream for shared expert multi-stream overlap.
        self._maybe_sync_shared_experts_stream(shared_experts_input)

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)

        with self._sequence_parallel_context():
            # TODO(bnell): parts of the dispatch/combine steps will go away once
            # #32567 lands and the remaining kernels are made MKs.  The PCP
            # code will probably remain
            hidden_states, router_logits = self._maybe_dispatch(
                hidden_states,
                router_logits,
            )

            shared_output, hidden_states = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
            )

            return self._maybe_combine(
                shared_output,
                hidden_states,
            )

    #########################################################
    #
    # Old methods from FusedMoE layer. Remove when possible.
    #
    #########################################################

    #
    # Properties
    #

    @property
    def layer_id(self):
        # Delayed import to avoid circular dependency
        from vllm.model_executor.models.utils import extract_layer_index

        return extract_layer_index(self.layer_name)

    #
    # Attributes still needed by models
    #

    @property
    def is_monolithic(self) -> bool:
        return self.routed_experts.quant_method.is_monolithic

    @property
    def activation(self) -> MoEActivation:
        return self.routed_experts.activation

    #
    # Expert maps
    #

    @property
    def expert_map_manager(self):
        """Forward to routed_experts.expert_map_manager for backward compatibility."""
        return self.routed_experts.expert_map_manager

    @property
    def expert_placement_strategy(self) -> ExpertPlacementStrategy:
        return self.expert_map_manager.placement_strategy

    @property
    def expert_global_to_physical(self) -> torch.Tensor | None:
        tables = self.expert_map_manager.routing_tables
        return tables[0] if tables else None

    @property
    def expert_physical_to_global(self) -> torch.Tensor | None:
        """Routing table: physical expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[1] if tables else None

    @property
    def expert_local_to_global(self) -> torch.Tensor | None:
        """Routing table: local expert ID to global expert ID."""
        tables = self.expert_map_manager.routing_tables
        return tables[2] if tables else None

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.routed_experts.expert_map

    def _expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        return self.routed_experts._expert_routing_tables()

    def update_expert_map(self):
        self.routed_experts.update_expert_map()

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        """Map global expert ID to local expert ID."""
        return self.routed_experts._map_global_expert_id_to_local_expert_id(expert_id)

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        return self.routed_experts.get_expert_weights()

    #
    # EPLB
    #

    @property
    def eplb_state(self) -> EplbLayerState | None:
        return self.router.eplb_state

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        if self.router.eplb_state is not None:
            self.router.eplb_state.set_layer_state(
                moe_layer_idx,
                expert_load_view,
                logical_to_physical_map,
                logical_replica_count,
            )
