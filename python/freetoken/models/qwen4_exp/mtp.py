"""Qwen3.8-Flash-Next single-block MTP draft head (DeepSeek nextn style).

Port of qwen3_5_moe/mtp.py for the hyper-connection stack: the draft layer is a
full qwen4_exp decoder layer — Qwen4ExpAttention (QSA) at layer_id = num_layers
writing its KV rows into the trunk pools, two GatedResidual hc blocks, and an
EAGER bf16 MoE (the checkpoint ships the MTP experts as one stacked bf16
``[E, ...]`` pair, unlike the trunk's per-layer NVFP4 banks; the draft runs
~1 token per step, so a chunked gather-bmm is the right shape). TP=1 only.

Carry and input fusion — the dual ``fc_hidden``/``fc_embedding`` neck (SGLang
``qwen4_exp_mtp._fuse_residual_linear_shared`` semantics)::

    e   = fc_embedding(pre_fc_norm_embedding(embed(ids)))       # [T, H]
    enc = fc_hidden(pre_fc_norm_hidden(carry).view(T, hc, H))   # [T, hc, H]
    in_ = (e.unsqueeze(-2) + enc).reshape(T, hc * H)            # e over all streams

The carry is the trunk's PRE-mix hyper-connection residual ``[T, hc*H]``
(``pre_fc_norm_hidden`` is hc-width), surfaced as ``last_hc_hidden`` /
``last_hidden`` by the qwen4_exp model. The head's top-level
``hyper_connection_mixer`` (hc_norm + mix, no injection) stands in for the
trunk's absent final norm before ``lm_head``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated

from .attention import Qwen4ExpAttention
from .hc import GatedResidual

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

# Gathered-expert rows per bmm slice: bounds the bf16 scratch at [rows, 2I, H]
# (16 rows ~= 208 MiB for the 512 x 1280 x 2560 MTP banks).
_EXPERT_CHUNK = 16


class _StackedExperts(BaseOP):
    """Checkpoint-shaped container: the stacked expert tensors are bare-tensor
    attributes (BaseOP emits no trailing ``.weight`` for them), matching the
    stripped remap keys in weight.py."""

    def __init__(self, num_experts: int, intermediate: int, hidden: int) -> None:
        self.gate_up_proj = torch.zeros(
            num_experts, 2 * intermediate, hidden, dtype=torch.bfloat16)
        self.down_proj = torch.zeros(
            num_experts, hidden, intermediate, dtype=torch.bfloat16)


class Qwen4ExpMTPMoE(BaseOP):
    """Eager routed MoE (512 experts, top-8, renormalized) + gated shared
    expert — Qwen4ExpMoE's math in plain torch over the stacked bf16 experts."""

    def __init__(self, config: ModelConfig, *, prefix: str) -> None:
        hidden = config.hidden_size
        self.hidden_size = hidden
        self.intermediate_size = config.moe_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = LinearReplicated(
            hidden, config.num_experts, has_bias=False, prefix=f"{prefix}.gate")
        self.experts = _StackedExperts(
            config.num_experts, self.intermediate_size, hidden)
        # Checkpoint-shaped submodule: keys are shared_expert.gate_up_proj /
        # shared_expert.down_proj, exactly the trunk MoE's layout.
        self.shared_expert = _SharedExpert(
            hidden, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert")
        self.shared_expert_gate = LinearReplicated(
            hidden, 1, has_bias=False, prefix=f"{prefix}.shared_expert_gate")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.gate.forward(hidden_states).float().softmax(dim=-1)
        topk_w, topk_i = torch.topk(scores, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        routed = self._routed(hidden_states, topk_i, topk_w)
        gate_up = self.shared_expert.gate_up_proj.forward(hidden_states)
        g, u = gate_up.chunk(2, dim=-1)
        shared = self.shared_expert.down_proj.forward(F.silu(g) * u)
        gate = torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        return routed + shared * gate

    def _routed(self, x: torch.Tensor, topk_i: torch.Tensor,
                topk_w: torch.Tensor) -> torch.Tensor:
        T, K = topk_i.shape
        H, I = self.hidden_size, self.intermediate_size
        gu_bank, dw_bank = self.experts.gate_up_proj, self.experts.down_proj
        out = x.new_zeros(T, H)
        flat_ids = topk_i.reshape(-1)  # token-major: slot j belongs to token j // K
        x_rows = x.repeat_interleave(K, dim=0)
        flat_w = topk_w.reshape(-1)
        # Accumulation rows are the SLOT position // K, never the expert id.
        rows = torch.arange(flat_ids.numel(), device=x.device) // K
        for s in range(0, flat_ids.numel(), _EXPERT_CHUNK):
            e = slice(s, s + _EXPERT_CHUNK)
            ids_e = flat_ids[e]
            if not gu_bank.is_cuda:
                # Host-resident stacks (the draft experts alone are ~4.7 GiB
                # bf16 -- more than this card's whole slack): index on the CPU
                # side, then one pinned H2D per chunk. The per-step slice is
                # topk rows ~= 78 MiB at B=1, a few ms over PCIe.
                ids_host = ids_e.cpu()
                gu = gu_bank[ids_host].to(x.device, non_blocking=True)
                dw = dw_bank[ids_host].to(x.device, non_blocking=True)
            else:
                gu = gu_bank[ids_e]  # [n, 2I, H]
                dw = dw_bank[ids_e]  # [n, H, I]
            xk = x_rows[e].unsqueeze(1)
            g = torch.bmm(xk, gu[:, :I].transpose(1, 2)).squeeze(1)
            u = torch.bmm(xk, gu[:, I:].transpose(1, 2)).squeeze(1)
            y = torch.bmm((F.silu(g) * u).unsqueeze(1), dw.transpose(1, 2)).squeeze(1)
            out.index_add_(0, rows[e],
                           y * flat_w[e].to(y.dtype).unsqueeze(1))
        return out


class _SharedExpert(BaseOP):
    def __init__(self, hidden: int, inter: int, *, prefix: str) -> None:
        self.gate_up_proj = LinearReplicated(
            hidden, 2 * inter, has_bias=False, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = LinearReplicated(
            inter, hidden, has_bias=False, prefix=f"{prefix}.down_proj")


class Qwen4ExpMTPDraftLayer(BaseOP):
    """One qwen4_exp decoder-layer clone (attention + hc blocks + eager MoE; no
    PLE). Forward contract identical to the trunk layers:
    ``R [T, hc*hidden] -> R' [T, hc*hidden]`` with immediate combines."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str) -> None:
        self.attn_hyper_connection = GatedResidual(
            config, prefix=f"{prefix}.attn_hyper_connection")
        self.self_attn = Qwen4ExpAttention(
            config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp_hyper_connection = GatedResidual(
            config, prefix=f"{prefix}.mlp_hyper_connection")
        self.mlp = Qwen4ExpMTPMoE(config, prefix=f"{prefix}.mlp")

    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        x, inject = self.attn_hyper_connection.mix(hidden)
        y = self.self_attn.forward(x, batch)
        hidden = self.attn_hyper_connection.combine(hidden, y, inject)
        x, inject = self.mlp_hyper_connection.mix(hidden)
        y = self.mlp.forward(x)
        return self.mlp_hyper_connection.combine(hidden, y, inject)


class Qwen4ExpMTPHead(BaseOP):
    """Single-block MTP draft head. spec_mtp.MTPDrafter feeds the trunk's
    pre-mix hc residual + the last sampled token; the KV rows land in the trunk
    pools at layer_id = num_layers (the extra storage layer parse_config folds
    into the QSA group)."""

    def __init__(self, config: ModelConfig, embed_tokens,
                 *, prefix: str = "model.mtp") -> None:
        hidden = config.hidden_size
        self.hc_count = config.qwen4_args.hc_count
        self.hidden_size = hidden
        self._embed_tokens = embed_tokens  # trunk-shared; _-prefixed: not in state dict
        self._lm_head = None
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(
            hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(
            self.hc_count * hidden, eps=config.rms_norm_eps)
        self.fc_embedding = LinearReplicated(
            hidden, hidden, has_bias=False, prefix=f"{prefix}.fc_embedding")
        self.fc_hidden = LinearReplicated(
            hidden, hidden, has_bias=False, prefix=f"{prefix}.fc_hidden")
        # The head's final norm: the trunk's top-level mixer shape (hc_norm +
        # mix, no injection) collapsing the hc streams before lm_head.
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        self.layer = Qwen4ExpMTPDraftLayer(
            config, config.num_layers, prefix=f"{prefix}.layer")

    def set_lm_head(self, lm_head) -> None:
        self._lm_head = lm_head

    @property
    def experts_host_resident(self) -> bool:
        return not self.layer.mlp.experts.gate_up_proj.is_cuda

    def offload_experts_to_host(self) -> None:
        """Pin the stacked draft experts in host RAM (~4.7 GiB bf16). The draft
        touches top-K expert rows per step (~78 MiB at B=1), so the PCIe gather
        in :meth:`Qwen4ExpMTPMoE._routed` costs a few ms -- far cheaper than
        the VRAM the stacks would hold on a 16 GiB card."""
        s = self.layer.mlp.experts
        for name in ("gate_up_proj", "down_proj"):
            t = getattr(s, name)
            if t.is_cuda:
                setattr(s, name, t.cpu().pin_memory())

    def forward_rows(self, carry: torch.Tensor, input_ids: torch.Tensor, *,
                     with_logits: bool):
        """One draft step: fuse embed+carry, run the draft layer, hc-mix.
        Returns ``(carry, logits)`` when ``with_logits`` else the carry; the
        caller samples and feeds the next carry."""
        T = carry.shape[0]
        batch = get_global_ctx().batch
        e = self._embed_tokens.forward(input_ids.reshape(-1))
        r_in = self._fuse_neck(e, carry)
        r_out = self.layer.forward(r_in, batch)
        if not with_logits:
            return r_out
        mixed, _ = self.hyper_connection_mixer.mix(r_out)
        return r_out, self._lm_head.forward(mixed)

    def _fuse_neck(self, embeds: torch.Tensor, carry: torch.Tensor) -> torch.Tensor:
        """Checkpoint input fusion (sglang ``_fuse_hc_input``): the normed
        embedding through ``fc_embedding``, the normed per-stream carry through
        ``fc_hidden``, summed per stream.
        ``[T, H] + [T, hc*H] -> [T, hc*H]``."""
        T = embeds.shape[0]
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embeds))
        enc = self.fc_hidden.forward(
            self.pre_fc_norm_hidden.forward(carry).reshape(-1, self.hidden_size))
        enc = enc.reshape(T, self.hc_count, self.hidden_size)
        return (e.unsqueeze(1) + enc).reshape(T, self.hc_count * self.hidden_size)

    def draft_step(self, carry: torch.Tensor, input_ids: torch.Tensor):
        """``(carry, logits)`` of the next draft position; spec_mtp samples and
        feeds the next carry."""
        return self.forward_rows(carry, input_ids, with_logits=True)


__all__ = ["Qwen4ExpMTPDraftLayer", "Qwen4ExpMTPHead", "Qwen4ExpMTPMoE"]
