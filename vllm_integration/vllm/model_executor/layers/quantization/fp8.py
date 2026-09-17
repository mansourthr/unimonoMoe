# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Literal

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    CutlassFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.ignored_layers_match_mode: Literal["exact", "substring", "suffix"] = (
            "exact"
        )
        self.store_dtype = store_dtype
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.use_deep_gemm: bool | None = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineLinearMethod,
                )

                online_method = Fp8PerTensorOnlineLinearMethod()
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.store_dtype == "mxfp4":
                from vllm.model_executor.layers.quantization.mxfp4 import (
                    Mxfp4MoEMethod,
                )

                return Mxfp4MoEMethod(layer.moe_config)
            if self.is_checkpoint_fp8_serialized:
                return Fp8MoEMethod(self, layer)
            else:
                from vllm.model_executor.layers.quantization.online.fp8 import (
                    Fp8PerTensorOnlineMoEMethod,
                )

                return Fp8PerTensorOnlineMoEMethod(layer=layer)
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    @staticmethod
    def get_cache_scale_mapper() -> "WeightsMapper":
        """Map compressed-tensors KV-cache scale names to vLLM names."""
        from vllm.model_executor.models.utils import WeightsMapper

        orig_to_new_suffix = {
            ".k_proj.output_scale": ".attn.k_scale",
            ".v_proj.output_scale": ".attn.v_scale",
            ".q_proj.output_scale": ".attn.q_scale",
            ".self_attn.prob_output_scale": ".self_attn.attn.prob_scale",
        }
        cache_scale_mapper = WeightsMapper(orig_to_new_suffix=orig_to_new_suffix)
        return cache_scale_mapper | QuantizationConfig.get_cache_scale_mapper()


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = False

        if self.quant_config.use_deep_gemm is not None:
            self.use_deep_gemm = self.quant_config.use_deep_gemm
        else:
            self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None

            self.activation_quant_key = create_fp8_quant_key(
                static=self.act_q_static,
                group_shape=GroupShape(1, self.weight_block_size[0]),
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
        else:
            self.weight_quant_key = kFp8StaticTensorSym
            # Use per-token quantization for better perf if dynamic and cutlass
            if self.act_q_static:
                self.activation_quant_key = kFp8StaticTensorSym
            elif cutlass_fp8_supported():
                self.activation_quant_key = kFp8DynamicTokenSym
            else:
                self.activation_quant_key = kFp8DynamicTensorSym

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if not self.block_quant:
            scale = create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
                scale_dtype=(torch.float8_e8m0fnu if self.is_scale_e8m0 else None),
            )
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_marlin:
            if not self.block_quant:
                # Canonicalize to (K, N) for the kernel.
                replace_parameter(layer, "weight", layer.weight.t())
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
            return

        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.block_quant:
            assert not self.act_q_static

        # If checkpoint not serialized fp8, quantize the weights.
        else:
            # If checkpoint is fp8 per-tensor, handle that there are N scales for N
            # shards in a fused module
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            # Update layer with new values.
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # if batch invariant mode is enabled, prefer direct FP8 path
        # we will use BF16 dequant when direct FP8 is not supported.
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.fp8_linear.apply_weights(
                    layer,
                    x,
                    bias,
                )
            else:
                if isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel):
                    return self.fp8_linear.apply_weights(layer, x, bias)

                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        return self.fp8_linear.apply_weights(layer, x, bias)


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        # Select Fp8 MoE backend
        # Whether this layer is eligible for the MoE monokernel fast path.
        # Decided after weight loading, in process_weights_after_loading.
        self._use_moe_monokernel = False

        # Captured here, not inside a forward: the fused TP gate needs values
        # that are identical on every rank, and VllmConfig is the one place they
        # are guaranteed to be. max_num_batched_tokens sizes the persistent
        # output pool; num_ubatches is how many MoE outputs can be live at once
        # and therefore how many pool slots there must be.
        _vc = get_current_vllm_config()
        self._mk_max_num_batched_tokens = int(
            _vc.scheduler_config.max_num_batched_tokens
        )
        self._mk_num_ubatches = int(_vc.parallel_config.num_ubatches)

        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    self.moe.w13_num_shards * intermediate_size_per_partition,
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=layer.orig_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # For per-tensor quant, the scales are per expert and weight.
            w13_scale_data = torch.ones(
                num_experts, self.moe.w13_num_shards, dtype=torch.float32
            )
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            # For block quant, the scales are per block (typically 128x128).
            w13_scale_data = torch.ones(
                num_experts,
                self.moe.w13_num_shards
                * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            )
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        # Note: name is weight_scale for tensor, weight_scale_inv for block.
        layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.fp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13,
                w13_scale,
                w13_input_scale,
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2,
                w2_scale,
                w2_input_scale,
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale,
                w2_input_scale,
                layer.moe_config.moe_parallel_config.enable_eplb,
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13,
                w13_scale,
                shard_size,
                layer.local_num_experts,
                is_act_and_mul=self.moe.is_act_and_mul,
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

        # Decide monokernel eligibility now that the weights are final. The
        # kernel consumes raw block-wise FP8 weights, so only the TRITON
        # backend qualifies; backends like DEEPGEMM repack them. The shape
        # test is the Qwen3.5 expert geometry the kernel is compiled for.
        # Gated by VLLM_USE_MOE_MONOKERNEL (default on); set it to 0 to force
        # the standard TRITON fused-MoE backend for A/B benchmarking.
        from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend

        self._use_moe_monokernel = False
        self._moe_monokernel_bias = None
        # True when this rank holds a strict subset of the global experts and the
        # loaded .so is the EP build. It selects the EP entry point at call time,
        # and it is decided once here rather than re-derived per call so the
        # decision that was validated is the decision that runs.
        self._moe_monokernel_ep = False
        eligible = (
            envs.VLLM_USE_MOE_MONOKERNEL
            and self.block_quant
            and self.fp8_backend == Fp8MoeBackend.TRITON
            and getattr(layer, "top_k", 1) > 1
        )
        if eligible:
            from vllm import _custom_ops as ops

            # Ask the binary what it was built for instead of hardcoding one
            # model's geometry. Every field the .so reports has to match the
            # layer, because the kernel indexes its weights from compile-time
            # constants: a shape it was not built for is read at the wrong
            # stride and still returns rc=0, which is a wrong answer rather
            # than a failure.
            try:
                geo = ops.prefill_monokernel_geometry()
            except Exception as exc:
                logger.info(
                    "MoE monokernel unavailable, could not read geometry "
                    "off the .so (%s)",
                    exc,
                )
                geo = None

            if geo is not None:
                want = {
                    "num_experts": getattr(layer, "global_num_experts", 0),
                    "top_k": getattr(layer, "top_k", 0),
                    "k_dim": layer.w13_weight.size(2),
                    "n_up": layer.w13_weight.size(1),
                    "n_half": layer.w2_weight.size(2),
                    "h_dim": layer.w2_weight.size(1),
                }
                # The .so reports ONE expert count, and it is used both to score
                # the router and to index the weight table, so it has to match
                # the global routing width AND the number of local weight slabs.
                # Those are the same number with EP off and differ under EP
                # (256 global against 32 local on an 8-rank node), and only the
                # global one was compared. A build at the EP width therefore
                # satisfied every checked field while being compiled to index
                # 256 slabs out of a 32-slab tensor, which is the wrong-stride
                # read the comment above warns about rather than a decline.
                local_experts = layer.w13_weight.size(0)
                want["num_experts_local"] = local_experts
                # A .so that cannot report a local count has no EP path: its
                # weight indexing is compiled against the global count. Mirroring
                # the global count here is what turns an EP-shaped weight tensor
                # into a MISMATCH instead of an unchecked assumption. A build that
                # DOES export the symbol reports its real value and is compared
                # against it.
                if (
                    geo.get("num_experts_local") is None
                    and geo.get("num_experts") is not None
                ):
                    geo = dict(geo)
                    geo["num_experts_local"] = geo["num_experts"]
                # A None means this .so predates the symbol, so it cannot be
                # checked and is not treated as a mismatch.
                mismatch = {
                    k: (geo.get(k), v)
                    for k, v in want.items()
                    if geo.get(k) is not None and geo[k] != v
                }

                # EXPERT PARALLELISM. Under EP the router still scores every
                # global expert while only this rank's slabs are resident, so the
                # kernel needs vLLM's own global->local map and a build whose
                # weight indexing, TMA extents and tile ownership are all local.
                # Both are checked HERE, at weight load, and not at the call:
                # once the persistent kernel is entered it holds every SM and its
                # grid barrier means one wrong rank wedges its peers instead of
                # failing alone. There is no safe post-entry fallback.
                ep_needed = local_experts != want["num_experts"]
                ep_capable = bool(geo.get("ep_capable") or 0)
                emap = getattr(layer, "expert_map", None) if ep_needed else None
                ep_reason = None
                if ep_needed and not ep_capable:
                    ep_reason = (
                        f"weights are EP shaped ({local_experts} local of "
                        f"{want['num_experts']} global) but the .so is not EP "
                        "capable"
                    )
                elif ep_capable and not ep_needed:
                    ep_reason = (
                        "the .so is an EP build but the weights hold every "
                        "global expert"
                    )
                elif ep_needed:
                    # Prove the tensor is the canonical -1/local-slot map rather
                    # than the 0/1 expert_mask the AITER kernels consume. The two
                    # have the same dtype and length, and reading the mask as a
                    # map would send every owned token to local slot 1.
                    if emap is None:
                        ep_reason = "layer exposes no expert_map"
                    elif emap.dtype != torch.int32:
                        ep_reason = f"expert_map dtype is {emap.dtype}, not int32"
                    elif emap.numel() != want["num_experts"]:
                        ep_reason = (
                            f"expert_map has {emap.numel()} entries, expected "
                            f"{want['num_experts']}"
                        )
                    else:
                        owned = int((emap >= 0).sum().item())
                        top = int(emap.max().item())
                        if owned != local_experts or top != local_experts - 1:
                            ep_reason = (
                                f"expert_map is not a global->local map: "
                                f"{owned} non-negative entries with max {top}, "
                                f"expected {local_experts} and "
                                f"{local_experts - 1}"
                            )
                if ep_reason is not None:
                    logger.info("MoE monokernel DECLINED: %s", ep_reason)
                    mismatch = dict(mismatch)
                    mismatch["ep"] = ep_reason

                # Scoring function. Mode 0 is softmax over the logits; modes 1
                # and 2 are sigmoid with a per-expert correction bias used for
                # selection only. Neither substitutes for the other, and the
                # difference is invisible in the weight shapes, so it is
                # checked explicitly rather than assumed.
                mode = geo.get("router_mode")
                scoring = getattr(layer, "scoring_func", "softmax")
                bias = getattr(layer, "e_score_correction_bias", None)
                if bias is not None and hasattr(bias, "data"):
                    bias = bias.data
                router_ok = True
                if (mode == 0 and scoring != "softmax") or (
                    mode in (1, 2) and scoring != "sigmoid"
                ):
                    router_ok = False
                elif mode in (1, 2) and bias is None:
                    # A sigmoid build with no bias would score every expert
                    # unbiased, which is not this model's routing either.
                    router_ok = False

                if mismatch or not router_ok:
                    logger.info(
                        "MoE monokernel DECLINED: geometry mismatch %s, "
                        "router_mode=%s scoring_func=%s bias=%s",
                        mismatch or "none",
                        mode,
                        scoring,
                        "present" if bias is not None else "absent",
                    )
                else:
                    self._use_moe_monokernel = True
                    self._moe_monokernel_bias = bias
                    self._moe_monokernel_ep = ep_needed
                    logger.info(
                        "MoE monokernel fast path ENABLED (E=%s, E_LOCAL=%s, "
                        "ep=%s, N_UP=%s, N_HALF=%s, K=%s, H=%s, top_k=%s, "
                        "router_mode=%s, scoring=%s, backend=%s)",
                        want["num_experts"],
                        local_experts,
                        int(ep_needed),
                        want["n_up"],
                        want["n_half"],
                        want["k_dim"],
                        want["h_dim"],
                        want["top_k"],
                        mode,
                        scoring,
                        self.fp8_backend,
                    )
                    # Build the fused TP plan HERE, while this is still ordinary
                    # eager Python on every rank. Its first call votes across the
                    # TP group and rendezvouses symmetric memory, and neither
                    # collective survives being reached later: a symmetric
                    # allocation cannot happen inside a CUDA graph capture, and
                    # the first reader of monokernel_output_is_reduced is
                    # MoERunner.forward on the Dynamo-traced side, which would
                    # trace the vote instead of running it. At this point the TP
                    # group is live, _setup_kernel has already run, and every
                    # rank arrives at the same layer in the same order. Later
                    # calls hit the module cache and take no collective.
                    if self._fused_tp_eligible():
                        from vllm.model_executor.layers.fused_moe.monokernel_tp import (  # noqa: E501
                            fused_tp_plan,
                        )

                        _tp_plan = fused_tp_plan(
                            self.moe.hidden_dim,
                            self._mk_max_num_batched_tokens,
                            self._mk_num_ubatches,
                        )
                        logger.info_once(
                            "fused TP MoE prefill plan built at weight load: "
                            "usable=%s fold_shared=%s reason=%s tokens=%s "
                            "slots=%s",
                            _tp_plan.usable,
                            _tp_plan.fold_shared,
                            _tp_plan.reason,
                            self._mk_max_num_batched_tokens,
                            _tp_plan.nslots,
                        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
            layer=layer,
        )

        # Inject biases into the quant config if the model has them
        # (e.g. GPT-OSS biased MoE)
        if quant_config is not None and self.moe.has_bias:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if w13_bias is not None:
                quant_config._w1.bias = w13_bias
            if w2_bias is not None:
                quant_config._w2.bias = w2_bias

        return quant_config

    @property
    def supports_eplb(self) -> bool:
        return True

    def _fused_tp_eligible(self) -> bool:
        """May this layer take the fused tensor-parallel epilogue at all?

        ONE PLACE, because the two callers must never disagree and the failure
        mode of disagreement is a double reduce: the epilogue hands back an
        output already summed across the group, so a caller that believes it is
        not reduced adds a second all-reduce and scales the activations by the
        world size. The callers are monokernel_output_is_reduced, which tells the
        runner the reduce is folded in, and try_moe_monokernel, which decides
        whether to enter the epilogue.

        RANK-UNIFORM BY CONSTRUCTION. Every input is either configuration, built
        once and handed to every worker, or the load-time monokernel
        eligibility, which is a geometry check over replicated weights. Nothing
        here is a function of the token count or of anything one rank observes
        alone.
        """
        if not getattr(self, "_use_moe_monokernel", False):
            return False
        # Mirrors the decline in MoERunner._maybe_moe_monokernel: when the
        # modular kernel runs the shared experts itself the monokernel is never
        # offered the work, so nothing on this path is reduced.
        if self.mk_can_overlap_shared_experts:
            return False
        # skip_final_all_reduce means the model reduces the MoE output itself
        # later, so entering the epilogue there would double-reduce.
        return self.moe.tp_size >= 2 and not self.moe.skip_final_all_reduce

    def monokernel_folds_shared_output(self) -> bool:
        """Does the fused epilogue also carry the shared expert output?

        Read by MoERunner._monokernel_folds_shared, which is the predicate that
        stops the runner reducing, scaling and adding the shared half a second
        time. SIZE-INDEPENDENT and RANK-UNIFORM for the same reasons as
        monokernel_output_is_reduced: the answer is the plan's fold_shared,
        which is one element of the same MIN all-reduce over the TP group that
        decided usable, so no rank can fold while a peer does not.
        """
        from vllm.model_executor.layers.fused_moe.monokernel_tp import (
            fused_tp_plan,
        )

        if not self._fused_tp_eligible():
            return False
        plan = fused_tp_plan(
            self.moe.hidden_dim,
            getattr(self, "_mk_max_num_batched_tokens", 0),
            getattr(self, "_mk_num_ubatches", 1),
        )
        return bool(plan.usable and plan.fold_shared)

    def monokernel_output_is_reduced(self) -> bool:
        """Does the monokernel path hand back an already TP-reduced output?

        SIZE-INDEPENDENT BY CONSTRUCTION, and that is the point rather than a
        simplification. MoERunner.forward reads this on the traced side, where
        the token count is symbolic and torch.compile will specialize the branch
        once. A per-size answer would be baked from the first size traced and
        then be wrong for every other one. So this says yes for the whole
        monokernel path and the reduce is folded in for the sizes the fused
        epilogue declines, instead of varying with M.

        RANK-UNIFORM BY CONSTRUCTION. tp_size, skip_final_all_reduce and the
        load-time monokernel eligibility all come from the config, which is
        built once and handed to every worker. plan.usable is the result of a
        MIN all-reduce over the TP group. The enable environment variable is
        never read here: it reaches this only through that vote.

        Returns False when skip_final_all_reduce is set, because there the model
        reduces the MoE output itself later and a fold would double-reduce.
        """
        from vllm.model_executor.layers.fused_moe.monokernel_tp import (
            fused_tp_plan,
        )

        if not self._fused_tp_eligible():
            return False
        plan = fused_tp_plan(
            # hidden_dim, not w2_weight.size(1): both are 7168 on DS3, and this
            # one is available without a layer and is the value the pool is
            # sized from everywhere else.
            self.moe.hidden_dim,
            getattr(self, "_mk_max_num_batched_tokens", 0),
            getattr(self, "_mk_num_ubatches", 1),
        )
        return bool(plan.usable)

    def try_moe_monokernel(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        residual: torch.Tensor | None = None,
        out_scale: float = 1.0,
    ) -> torch.Tensor | None:
        """Run the standalone prefill MoE monokernel, or return None to decline.

        Eligibility was decided at weight load time in
        process_weights_after_loading, where the expert geometry and the chosen
        FP8 backend are both final. The remaining test here is the token count,
        which has to be per call: the kernel serves a batch only up to the tile
        cap its binary was built for, and above that the standard path is
        correct and this returns None. A launch failure is treated the same way,
        so a fallback is always available and never a silent wrong answer.

        The fused tensor-parallel path is the one exception to that last
        sentence and is therefore kept OUT of the try/except below. Its epilogue
        is a collective: a rank that caught an error and returned None would
        leave its peers spinning on a barrier no one will ever arrive at. Its
        eligibility is agreed across the TP group before entry instead, and once
        entered it does not fall back. See fused_moe/monokernel_tp.py.
        """
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.monokernel_tp import (
            fused_tp_launch,
        )

        if not getattr(self, "_use_moe_monokernel", False):
            return None
        if x.size(0) > ops.MOE_MONOKERNEL_PREFILL_MAX_TOKENS:
            return None

        # The same predicate the fold reads. Entering the epilogue while the
        # runner believes the output is unreduced, or the reverse, is a wrong
        # answer rather than a slow one, so there is only one of it.
        fused = (
            fused_tp_launch(
                x,
                router_logits,
                layer,
                self.weight_scale_name,
                self._moe_monokernel_bias,
                # getattr, not attribute access: a quant method built outside a
                # VllmConfig context has neither, and 0 tokens makes the gate
                # refuse rather than raise.
                getattr(self, "_mk_max_num_batched_tokens", 0),
                getattr(self, "_mk_num_ubatches", 1),
                # Only the fused epilogue can carry these. The ordinary
                # monokernel below is left exactly as it was, so a token count
                # the epilogue declines produces an unfolded partial output and
                # the runner combines the halves itself.
                residual,
                out_scale,
            )
            if self._fused_tp_eligible()
            else None
        )
        if fused is not None:
            return fused

        try:
            return torch.ops.vllm.moe_monokernel_prefill(
                x,
                router_logits,
                layer.w13_weight,
                getattr(layer, f"w13_{self.weight_scale_name}"),
                layer.w2_weight,
                getattr(layer, f"w2_{self.weight_scale_name}"),
                layer.top_k,
                getattr(layer, "renormalize", True),
                self._moe_monokernel_bias,
                # None under TP, which is the shipped call unchanged. Under EP
                # this is vLLM's own global->local map, already validated against
                # the binary at weight load; the kernel returns this rank's
                # PARTIAL sum and MoERunner performs the cross-rank reduce
                # because MoEPrepareAndFinalizeNoDPEPModular.output_is_reduced()
                # is False.
                layer.expert_map if self._moe_monokernel_ep else None,
            )
        except RuntimeError as e:
            logger.warning_once(
                "prefill monokernel failed (%s); falling back to Triton", e
            )
            return None

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None

        fused_out = self.try_moe_monokernel(layer, x, router_logits)
        if fused_out is not None:
            return fused_out

        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
