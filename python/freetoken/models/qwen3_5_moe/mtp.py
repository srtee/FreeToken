"""Qwen3.5/3.6 single-block MTP draft head (DeepSeek-V3 nextn style).

Wave 0 of docs/mtp-plan.md: weights + module + eager numerics. The draft
block reuses the trunk's embedding table and lm_head by reference
(``mtp_use_dedicated_embeddings: false`` — checkpoint has no MTP embedding
or head tensors, matching buun's shared sidecar mode).

Semantics ported 1:1 from buun's ``graph_mtp`` (src/models/qwen35moe.cpp):
    e = enorm(embd); h = hnorm(h_prev)
    x = eh_proj(cat(e, h))                 # checkpoint key: mtp.fc
    x = x + o_proj(attn(q,k,v) * sigmoid(gate))
    x = moe(post_norm(x)) + x
    carry = norm(x)                        # -> lm_head / next draft step
All norms are Gemma-style (1 + weight) — the arch-wide convention the
weight loader bakes in. Attention: per-head q|gate halves (buun views the
q_proj output as (head_dim, 2) per head), q/k RMSNorm, partial NeoX rope,
GQA against the trunk's 2 kv heads.

State-dict keys mirror the loader's emitted names 1:1 (mtp.* -> model.mtp.*,
mtp.layers.0.* -> model.mtp.layer.*): the fused stacked experts keep their
``experts.gate_up_proj`` / ``experts.down_proj`` names (no trailing
``.weight``), matching the checkpoint.

Wave-0 attention is a self-contained eager path (no paged KV): it attends
over an optional explicit KV list or just itself (single-token numerics).
Wave 1 re-points it at the trunk pool via layer_id = num_layers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from freetoken.layers.base import BaseOP
from freetoken.layers.norm import GemmaRMSNorm
from .attention import Qwen3_5Attention


def _rope_neox(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int,
               base: float) -> torch.Tensor:
    """Partial NeoX rope over the first ``rotary_dim`` dims of x ([T, H, D])."""
    half = rotary_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32,
                                       device=x.device) / half))
    # x: [T, H, D]; positions: [T] -> angles [T, 1, half] broadcasts over heads
    angles = positions.to(torch.float32)[:, None, None] * inv[None, None, :]
    cos, sin = angles.cos(), angles.sin()
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    out_rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([out_rot, x_pass], dim=-1).to(x.dtype)


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm with the (1 + w) bake already stored in ``weight``
    (the loader adds 1 to the checkpoint's raw norm weights, mirroring the
    trunk path — GemmaRMSNorm applies the stored weight directly)."""
    v = x
    out = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return weight * out


class MTPMoE(BaseOP):
    """Eager routed MoE (256 experts, top-8, renormalized) + gated shared
    expert — the same math as Qwen3_5MoE in plain torch. The draft runs 1
    token per step, so a gather-bmm over the stacked bf16 experts is the
    right shape; the fused stacked layout matches the checkpoint's
    mtp.layers.0.mlp.experts.{gate_up_proj,down_proj} names."""

    def __init__(self, config, hidden: int, inter: int, dtype=torch.bfloat16):
        self.top_k = config.num_experts_per_tok
        from freetoken.layers import LinearReplicated
        self.gate = LinearReplicated(hidden, config.num_experts, has_bias=False,
                                     prefix="model.mtp.layer.mlp.gate")
        E = config.num_experts
        class _Experts(BaseOP):
            pass
        self.experts = _Experts()
        self.experts.gate_up_proj = torch.zeros(E, 2 * inter, hidden, dtype=dtype)
        self.experts.down_proj = torch.zeros(E, hidden, inter, dtype=dtype)
        class _Shared(BaseOP):
            pass
        self.shared_expert = _Shared()
        self.shared_expert.gate_proj = LinearReplicated(
            hidden, inter, has_bias=False, prefix="model.mtp.layer.mlp.shared_expert.gate_proj")
        self.shared_expert.up_proj = LinearReplicated(
            hidden, inter, has_bias=False, prefix="model.mtp.layer.mlp.shared_expert.up_proj")
        self.shared_expert.down_proj = LinearReplicated(
            inter, hidden, has_bias=False, prefix="model.mtp.layer.mlp.shared_expert.down_proj")
        self.shared_expert_gate = LinearReplicated(
            hidden, 1, has_bias=False, prefix="model.mtp.layer.mlp.shared_expert_gate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.gate.forward(x).softmax(dim=-1)
        top_w, top_i = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
        gu = self.experts.gate_up_proj[top_i.reshape(-1)]
        d = self.experts.down_proj[top_i.reshape(-1)]
        xe = x.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, x.shape[-1])
        h = torch.bmm(gu, xe.unsqueeze(-1)).squeeze(-1)
        g, u = h.chunk(2, dim=-1)
        act = F.silu(g) * u
        out = torch.bmm(d, act.unsqueeze(-1)).squeeze(-1)
        out = (out.view(-1, self.top_k, x.shape[-1])
               * top_w.unsqueeze(-1)).sum(1)
        shared = (F.silu(self.shared_expert.gate_proj.forward(x))
                  * self.shared_expert.up_proj.forward(x))
        shared = self.shared_expert.down_proj.forward(shared)
        gate = torch.sigmoid(self.shared_expert_gate.forward(x))
        return out + shared * gate


class MTPDraftLayer(BaseOP):
    """The draft block's decoder layer: a TRUNK Qwen3_5Attention (paged KV,
    layer_id = num_layers — the draft's rows live in the trunk pool at the
    extra storage layer) + the eager MTPMoE. Keys mirror
    mtp.layers.0.{self_attn,mlp,input_layernorm,post_attention_layernorm}."""

    def __init__(self, config: ModelConfig, layer_id: int, dtype=torch.bfloat16):
        self.self_attn = Qwen3_5Attention(
            config, layer_id, prefix="model.mtp.layer.self_attn")
        self.mlp = MTPMoE(config, config.hidden_size,
                          config.moe_intermediate_size, dtype)
        self.eps = config.rms_norm_eps
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = _rms(x, self.input_layernorm, self.eps)
        x = self.self_attn.forward(x)
        x = x + residual
        x = self.mlp.forward(_rms(x, self.post_attention_layernorm, self.eps))
        return x + residual


class MTPHead(BaseOP):
    """Single-block MTP draft head. The draft step (spec_mtp.MTPDrafter)
    feeds the trunk's post-norm carry + the last sampled token; the KV rows
    land in the trunk pool at layer_id = num_layers."""

    def __init__(self, config: ModelConfig, embed_tokens,
                 *, prefix: str = "model.mtp", dtype=torch.bfloat16):
        hidden = config.hidden_size
        self.config = config
        self._embed_tokens = embed_tokens
        self._lm_head = None  # attached post-construction (shared trunk head)
        self.eps = config.rms_norm_eps
        layer_id = config.num_layers  # KV rows at the extra storage layer
        # GemmaRMSNorm/Linear give the state-dict the .weight suffix the
        # loader emits (trunk convention)
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        from freetoken.layers import LinearReplicated
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False,
                                   prefix="model.mtp.fc")
        self.norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.layer = MTPDraftLayer(config, layer_id, dtype)

    def set_lm_head(self, lm_head) -> None:
        self._lm_head = lm_head

    def draft_step(self, last_hidden_normed: torch.Tensor,
                   input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One draft step: embed+carry -> fc -> layer -> norm. Returns
        (carry, logits). The caller samples and feeds the next carry."""
        ids = input_ids.reshape(-1)
        e = self.pre_fc_norm_embedding.forward(self._embed_tokens(ids))
        h = self.pre_fc_norm_hidden.forward(last_hidden_normed)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x = self.layer.forward(x)
        carry = self.norm.forward(x)
        return carry, self._lm_head.forward(carry)


class MTPAttention(BaseOP):
    """Wave-0 eager attention (no paged KV) — kept for the unit-level
    numerics tests; production drafting uses MTPDraftLayer's trunk
    Qwen3_5Attention."""

    def __init__(self, config, hidden: int, dtype=torch.bfloat16):
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_config.rotary_dim
        self.rope_base = config.rotary_config.base
        self.eps = config.rms_norm_eps
        hd, hq, hv = self.head_dim, self.num_q, self.num_kv
        self.q_proj = torch.zeros(hq * hd * 2, hidden, dtype=dtype)
        self.k_proj = torch.zeros(hv * hd, hidden, dtype=dtype)
        self.v_proj = torch.zeros(hv * hd, hidden, dtype=dtype)
        self.o_proj = torch.zeros(hidden, hq * hd, dtype=dtype)
        self.q_norm = torch.zeros(hd, dtype=dtype)
        self.k_norm = torch.zeros(hd, dtype=dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                kv: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        xf = x
        T = x.shape[0]
        qg = (xf @ self.q_proj.T).view(T, self.num_q, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        k = (xf @ self.k_proj.T).view(T, self.num_kv, self.head_dim)
        v = (xf @ self.v_proj.T).view(T, self.num_kv, self.head_dim)
        q = _rms(q, self.q_norm, self.eps)
        k = _rms(k, self.k_norm, self.eps)
        q = _rope_neox(q, positions, self.rotary_dim, self.rope_base)
        k = _rope_neox(k, positions, self.rotary_dim, self.rope_base)
        if kv is not None:
            kk = torch.cat([kv[0], k], dim=0)
            vv = torch.cat([kv[1], v], dim=0)
        else:
            kk, vv = k, v
        rep = self.num_q // self.num_kv
        kk = kk.repeat_interleave(rep, dim=1)
        vv = vv.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, kk) * self.head_dim ** -0.5
        attn = scores.softmax(dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, vv).reshape(T, -1)
        out = out * torch.sigmoid(gate.reshape(T, -1))
        return out @ self.o_proj.T


