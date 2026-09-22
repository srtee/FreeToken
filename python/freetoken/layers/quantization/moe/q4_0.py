"""GGUF Q4_0 experts (the ``expert_quant="q4_0"`` format tag): native ggml block
bytes served from the offload cache through the borrowed ggml MoE kernels.

The banks are the checkpoint's own packed blocks -- gate/up Q4_0, the down
projection normalized to Q4_1 (the per-bank ggml type rides in
``cache.ggml_types``) -- so there is no repack: ``pack`` echoes its inputs and
``layout`` mirrors the offload cache's ``q4_0`` bank schema. Forward is the
MMVQ vector kernel for both prefill and decode (``topk_ids`` index cache slots
/ layer positions directly).
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch


from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod

# ggml Q4_0 block: 32 values -> 18 packed bytes ([d:fp16][16B nibbles]);
# Q4_1 adds an [m:fp16] min per block -> 20 bytes. The down bank is ALWAYS
# Q4_1: both providers normalize it there (qwen3moe upconverts its mixed
# Q4_0/Q4_1 down tensors, bit-exact with m = fp16(-8*d)), and the ggml MoE
# kernel wants Q4_1 for the down GEMV.
_BLOCK, _BYTES = 32, 18
_BYTES_Q4_1 = 20


class GgmlQ4_0MoEKernel(MoEKernel):
    """Dequant-in-kernel grouped GEMV over native Q4_0 / Q4_1 banks."""

    name: ClassVar[str] = "ggml"
    cpu_format: ClassVar[str | None] = "q4_0"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        from freetoken.moe.fused_q4_0 import _ACT

        reason = self._common_reject(cfg, resident_ok=False, tp_ok=True, cpu_ok=True, plain_silu_only=False)
        if reason is not None:
            return reason
        if cfg.interleaved:
            return "the ggml MoE kernels read uninterleaved [gate; up] rows"
        if cfg.activation not in _ACT:
            return f"no {cfg.activation!r} epilogue in the ggml MoE kernels"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, ((h + _BLOCK - 1) // _BLOCK) * _BYTES), torch.uint8),
            "down": BankSpec((h, ((i + _BLOCK - 1) // _BLOCK) * _BYTES_Q4_1), torch.uint8),
        }

    def pack(self, pieces: dict[str, torch.Tensor], cfg: MoEConfig, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out.update(pieces)  # the packed blocks are stored verbatim
        return {}
    def apply(self, layer: Any, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, view: ExpertView, *, is_prefill: bool) -> torch.Tensor:
        from freetoken.models.gguf.dequant import GGML_Q4_1
        from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0
        return fused_experts_gguf_q4_0(
            x, view.tensors["gate_up"], view.tensors["down"], topk_weights, topk_ids,
            layer.activation, down_qt=(layer.offload_cache.ggml_types or {}).get("down", GGML_Q4_1),
        )


@register_method(QuantKind.Q4_0, LayerKind.MOE)
class Q4_0MoEMethod(MoEMethod):
    candidates = (GgmlQ4_0MoEKernel,)

    def create_weights(self, layer: Any) -> None:
        raise NotImplementedError("Q4_0 experts are served from the offload cache, not resident")

    def resident_view(self, layer: Any) -> ExpertView:
        raise NotImplementedError("Q4_0 experts are not resident")
