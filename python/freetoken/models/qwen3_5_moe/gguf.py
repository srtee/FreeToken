"""Qwen3.5/3.6-MoE (arch ``qwen35moe``) GGUF adapter: build the FreeToken
``ModelConfig`` from GGUF metadata and map GGUF tensors onto the
``qwen3_5_moe`` modules.

This family is the hybrid GDN + gated-attention MoE line (Qwen3-Next style):
of ``block_count`` layers, every ``full_attention_interval``-th (1-indexed) is
a full-attention layer and the rest are GatedDeltaNet linear layers. The GGUF
layout (unsloth NVFP4 build):

- Full-attention blocks: ``attn_q`` [2q, H] (the q half includes the 2x
  output-gate rows), ``attn_k``/``attn_v`` [kv], ``attn_output``, q/k norms.
  q|k|v fuse into the model's ``qkv_proj`` (split [2q, kv, kv]).
- Linear blocks: ``attn_qkv`` [conv_dim, H] (q|k|v of the in-proj), ``attn_gate``
  [value_dim, H] (z), ``ssm_beta``/``ssm_alpha`` [dt_rank, H] (b/a),
  ``ssm_a`` [dt_rank] (A_log), ``ssm_dt.bias`` [dt_rank], ``ssm_norm`` [head_v],
  ``ssm_out`` [H, value_dim], ``ssm_conv1d`` [conv_dim, K]. All fuse into the
  model's ``in_proj.weight`` in split order [conv_dim, value_dim, b, a].
- Every block: MoE ``ffn_{gate,up,down}_exps`` (NVFP4, routed experts) +
  ``ffn_{gate,up,down}_shexp`` (NVFP4, gated shared expert) +
  ``ffn_gate_inp`` (F32 router) + ``ffn_gate_inp_shexp`` (F32 shared gate).
- ``token_embd``/``output``/norms: BF16.

Dense NVFP4 tensors dequantize to bf16 (GGUF checkpoints get
``config.quant = None``, so the model's linears are plain bf16): the GGML
NVFP4 block layout (4 ue4m3 scale bytes + 32 nibble bytes per 64 elements)
decodes as ``e2m1(nibble) * e4m3(scale_byte) * tensor_global`` where the
per-tensor global comes from the sibling ``.scale`` side tensor -- the ggml
doubled-kvalues convention and the ue4m3 0.5 factor cancel exactly.

Routed experts stay packed: they convert to the offload cache's native
``"nvfp4"`` bank layout (packed nibble bytes + fp8 block-scale bytes + fp16
per-row globals) in :func:`load_nvfp4_expert_sources`, so the ~18 GB expert
payload never expands to bf16 on a 30 GB host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 loudly."""
    from freetoken.distributed import get_tp_info

    tp = get_tp_info()
    if tp.size != 1:
        raise ValueError(f"{what} requires TP=1 (got TP={tp.size})")


# GGML NVFP4 (type 40) block layout: 64 elements = 4 sub-blocks of 16, each
# sub-block = 1 ue4m3 scale byte + 8 nibble bytes; one super-block = 36 bytes.
_NVFP4_BLOCK = 64
_NVFP4_BYTES = 36

# E2M1 value table (FreeToken kernel convention: value = table[code]).
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_E2M1_LUT = torch.tensor(_E2M1_VALUES, dtype=torch.float32)


def _e4m3_u8_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode unsigned e4m3 scale bytes (bias 7). ggml's ue4m3 never sets the
    sign bit, so this is plain fp8-e4m3 decode of the positive half."""
    x = x.to(torch.int32) & 0xFF
    e = (x >> 3) & 0xF
    m = (x & 7).to(torch.float32)
    normal = (1.0 + m / 8.0) * torch.exp2((e - 7).to(torch.float32))
    sub = m / 8.0 * 2.0 ** -6
    return torch.where(e == 0, sub, normal)


def _dequant_nvfp4_rows(raw: torch.Tensor, row_global: torch.Tensor | None) -> torch.Tensor:
    """Dequantize GGML NVFP4 packed rows ``[rows, row_bytes]`` (row_bytes a
    multiple of 36) to ``[rows, cols]`` float32, applying the optional per-row
    or broadcast global scale."""
    rows, row_bytes = raw.shape
    sb = raw.reshape(rows, row_bytes // _NVFP4_BYTES, _NVFP4_BYTES)
    n_sb = sb.shape[1]
    scales = _e4m3_u8_to_f32(sb[:, :, :4])  # [rows, n_sb, 4]
    nibbles = sb[:, :, 4:]                  # [rows, n_sb, 32]
    lo_v = _E2M1_LUT[(nibbles & 0xF).long()]  # [rows, n_sb, 32] -> LUT
    hi_v = _E2M1_LUT[(nibbles >> 4).long()]
    # nibble byte g of a super-block: sub-block r = g//8, position i = g%8.
    # sub-block r holds super-block elements 16*r..16*r+15: lo nibble = elem
    # 16*r+i, hi nibble = elem 16*r+8+i.
    lo_v = lo_v.view(rows, n_sb, 4, 8)
    hi_v = hi_v.view(rows, n_sb, 4, 8)
    out = torch.empty(rows, n_sb, 64, dtype=torch.float32)
    out[:, :, 0:8] = lo_v[:, :, 0, :]; out[:, :, 8:16] = hi_v[:, :, 0, :]
    out[:, :, 16:24] = lo_v[:, :, 1, :]; out[:, :, 24:32] = hi_v[:, :, 1, :]
    out[:, :, 32:40] = lo_v[:, :, 2, :]; out[:, :, 40:48] = hi_v[:, :, 2, :]
    out[:, :, 48:56] = lo_v[:, :, 3, :]; out[:, :, 56:64] = hi_v[:, :, 3, :]
    flat = out.reshape(rows, n_sb * _NVFP4_BLOCK)
    # each sub-block (16 elems) shares one scale: sb scale byte r covers
    # elements 16r..16r+15.
    flat = flat * scales.reshape(rows, n_sb, 4, 1).expand(-1, -1, -1, 16).reshape(rows, -1)
    if row_global is not None:
        flat = flat * row_global.reshape(-1, 1).to(torch.float32)
    return flat


# Side-scale cache shared between the weight iterator and the bank loader.
# The reader streams tensors once per call, so each pass re-reads the side
# scales it needs; this dict is populated by both entrypoints.
_side_scales: dict[str, dict[str, torch.Tensor]] = {}


def _side_scale(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def _to_bf16(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _to_f32(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def _nvfp4_global_for(t, layer_sides: dict) -> torch.Tensor | None:
    """Per-row global scale for an NVFP4 weight tensor: the sibling ``.scale``
    is per-tensor ``[1]`` (broadcast) or per-expert ``[E]`` (shared by every
    row of one expert). Returns ``[rows]`` fp32, or None when absent."""
    g = layer_sides.get("scale")
    if g is None:
        return None
    rows = t.rows
    if g.numel() == rows:
        return g.reshape(-1)
    if g.numel() == 1:
        return g.reshape(1).expand(rows)
    per = rows // g.numel()
    return g.reshape(-1, 1).expand(-1, per).reshape(-1)


def _dense_tensor(t, layer_sides: dict) -> torch.Tensor:
    """Dense projection -> bf16, dispatched on the tensor's ggml type:
    NVFP4 (40) block-decodes with its per-tensor global; BF16 (30) and F32 (0)
    pass through (the model downcasts bf16 weights at load)."""
    if t.ggml_type == 40:  # GGMLQuantizationType.NVFP4
        global_scale = _nvfp4_global_for(t, layer_sides)
        out = _dequant_nvfp4_rows(t.packed().clone(), global_scale)
        return out.reshape(t.shape).to(torch.bfloat16)
    if t.ggml_type == 30:  # BF16
        return _to_bf16(t)
    if t.ggml_type == 0:   # F32
        return _to_bf16(t)
    raise ValueError(f"unexpected ggml type {t.ggml_type} for dense tensor")


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata
    arch = shim.model_type  # "qwen35moe"

    def g(key: str):
        val = m.get(f"{arch}.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key {arch}.{key}")
        return val

    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    kv_heads = g("attention.head_count_kv")
    num_kv_heads = int(kv_heads[0] if isinstance(kv_heads, (list, tuple)) else kv_heads)
    head_dim = int(g("attention.key_length"))
    max_pos = int(g("context_length"))
    num_layers = int(g("block_count"))
    interval = int(g("full_attention_interval"))
    ssm_k_heads = int(g("ssm.group_count"))
    ssm_state = int(g("ssm.state_size"))          # key & value head dim
    ssm_inner = int(g("ssm.inner_size"))          # value_dim = v_heads * state
    conv_kernel = int(g("ssm.conv_kernel"))

    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    linear_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
    num_v_heads = ssm_inner // ssm_state

    # Partial rotary: llama.cpp writes rope.dimension_count = rotary_dim
    # (qwen3.5 full-attn rotary = 64 of 256 head dims; the mRoPE sections
    # [11,11,10,0] reduce to standard 1D rope for text-only serving since
    # all three position streams coincide).
    rotary_dim = int(g("rope.dimension_count")) if f"{arch}.rope.dimension_count" in m else head_dim
    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=float(g("rope.freq_base")),
        scaling=None,
    )
    full_group = FullAttentionGroupConfig(
        name="full",
        layer_ids=full_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_config=rotary,
    )
    linear_group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=linear_ids,
        num_key_heads=ssm_k_heads,
        num_value_heads=num_v_heads,
        key_head_dim=ssm_state,
        value_head_dim=ssm_state,
        conv_kernel_dim=conv_kernel,
        output_gate="silu",
    )
    groups = tuple(sorted((full_group, linear_group), key=lambda grp: grp.layer_ids[0]))

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=rotary,
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        shared_expert_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type=arch,
        architectures=list(shim.architectures),
        attention_groups=groups,
        # Routed experts keep their packed NVFP4 through the offload banks;
        # dense tensors dequantize to bf16 (GGUF -> config.quant=None).
        expert_quant="nvfp4",
        moe_weight_format="nvfp4",
        weight_format="gguf",
        gguf_type_table={},
    )


# Per-layer norm tensors (gguf suffix -> freetoken module-relative name). The
# layer norms are Gemma-style (1 + weight): the loader bakes the +1 in, so the
# adapter adds 1 here.
_LAYER_NORM_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

# Routed-expert tensors are owned by the NVFP4 bank loader, skipped here.
_EXPERT_SKIP_PREFIXES = (
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
)


def _shim_for(model_path: str) -> "GgufConfigShim":
    from freetoken.models.gguf.config import build_gguf_shim

    return build_gguf_shim(model_path)


def _collect_side_scales(model_path: str) -> dict[str, dict[str, torch.Tensor]]:
    """One pass over the file collecting the ``.scale``/``.input_scale`` side
    tensors (per-tensor or per-expert globals) keyed by their weight name."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    out: dict[str, dict[str, torch.Tensor]] = {}
    for t in iter_gguf_tensors(model_path):
        suffix = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        if suffix.endswith(".scale") or suffix.endswith(".input_scale"):
            # key by the weight tensor's full name: base + ".weight".
            base = t.name[: -len(".scale")] if suffix.endswith(".scale") else t.name[: -len(".input_scale")]
            wname = base + ".weight"
            out.setdefault(wname, {})[
                "scale" if suffix.endswith(".scale") else "input_scale"
            ] = _side_scale(t)
    return out


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen35moe param, bf16.

    Dense NVFP4 projections dequantize to bf16 (GGUF -> ``config.quant=None``).
    Routed experts are skipped (the NVFP4 bank provider owns them); the router
    and the shared-expert gate dequantize to bf16/f32.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(_shim_for(model_path))
    # GQA head-order fix: the checkpoint's GDN q/k/v rows pair v-head m
    # with k/q-head m % num_k_heads (block layout, matching llama.cpp's
    # fused GDN kernel), while the engine's fla kernels pair v-head j
    # with k/q-head j // (num_v_heads // num_k_heads) (interleave).
    # Permute every v-head-indexed weight segment by pi so the interleave
    # kernels read the same data llama.cpp reads: ft v-head j loads
    # GGUF v-head pi(j), pi(j) = j//g + num_k_heads*(j%g). Verified
    # exact vs llama.cpp layer-0 GDN activations on the real checkpoint.
    _lg = config.linear_attention_group()
    g = _lg.num_value_heads // _lg.num_key_heads
    _pi = torch.tensor(
        [(j // g) + _lg.num_key_heads * (j % g) for j in range(_lg.num_value_heads)]
    )

    def _perm_v(t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Gather the v-head-indexed ``dim`` of ``t`` by pi. The dim is
        either ``num_v_heads`` scalars (per-head params) or
        ``num_v_heads * head_v_dim`` (head-blocked segments)."""
        hv = _lg.num_value_heads
        n = t.shape[dim]
        if n == hv:
            return t.index_select(dim, _pi)
        d = n // hv
        if hv * d != n:
            return t  # not v-head-indexed (e.g. dt-rank params); leave as-is
        moved = t.transpose(dim, -1).contiguous()       # [..., hv, d] -> last
        reshaped = moved.reshape(-1, hv, d)
        out = reshaped.index_select(1, _pi).reshape(moved.shape)
        return out.transpose(dim, -1).contiguous().reshape(t.shape)

    # GDN in-proj fusion buffers (linear layers): qkv + z + b + a rows concat
    # into ``linear_attn.in_proj.weight`` in the model's split order
    # [conv_dim(qkv), value_dim(z), b, a].
    in_proj_buf: dict[int, dict[str, torch.Tensor]] = {}
    # Shared-expert gate/up fusion buffers.
    shexp_buf: dict[int, dict[str, torch.Tensor]] = {}
    # Full-attn qkv fusion buffers.
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    full_ids = {
        layer_id for layer_id in range(config.num_layers)
        if not config.is_linear_layer(layer_id)
    }
    # Side scales may stream AFTER their weight tensor; pre-read them so the
    # NVFP4 dequant has the global available at weight time.
    sides = _collect_side_scales(model_path)

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    def flush_in_proj(layer: int, base: str):
        slots = in_proj_buf.pop(layer)
        yield f"{base}.linear_attn.in_proj.weight", torch.cat(
            [slots["qkv"], slots["z"], slots["b"], slots["a"]], dim=0
        )

    def flush_shexp(layer: int, base: str):
        slots = shexp_buf.pop(layer)
        yield f"{base}.mlp.shared_expert.gate_up_proj.weight", torch.cat(
            [slots["gate"], slots["up"]], dim=0
        )

    def flush_qkv(layer: int, base: str):
        slots = qkv_buf.pop(layer)
        yield f"{base}.self_attn.qkv_proj.weight", torch.cat(
            [slots["q"], slots["k"], slots["v"]], dim=0
        )

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.weight", _to_bf16(t)
            continue
        if name == "output.weight":
            yield "lm_head.weight", _to_bf16(t)
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)  # (1+w_hf) pre-baked
            continue
        if not name.startswith("blk."):
            continue

        layer = layer_of(name)
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"
        is_full = layer in full_ids

        if suffix.endswith(".scale") or suffix.endswith(".input_scale"):
            wname = name[: -len(".scale")] if suffix.endswith(".scale") else name[: -len(".input_scale")]
            sides.setdefault(wname, {})[
                "scale" if suffix.endswith(".scale") else "input_scale"
            ] = _side_scale(t)
            continue

        if suffix in _LAYER_NORM_MAP and (is_full or suffix.startswith(("attn_norm", "post_attention"))):
            yield f"{base}.{_LAYER_NORM_MAP[suffix]}", _to_bf16(t)  # (1+w_hf) pre-baked
            continue
        if any(suffix.startswith(p) for p in _EXPERT_SKIP_PREFIXES):
            continue  # routed experts -> NVFP4 bank provider

        if is_full:
            if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                part = {"attn_q.weight": "q", "attn_k.weight": "k", "attn_v.weight": "v"}[suffix]
                qkv_buf.setdefault(layer, {})[part] = _dense_tensor(t, sides.get(name, {}))
                if all(k in qkv_buf[layer] for k in ("q", "k", "v")):
                    yield from flush_qkv(layer, base)
            elif suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.weight", _dense_tensor(t, sides.get(name, {}))
            elif suffix.startswith("ffn_") or suffix.startswith(("attn_norm", "post_attention")):
                pass  # handled by the shared MoE dispatch below
            else:
                raise ValueError(f"unmapped qwen35moe full-attn tensor: {name}")
        else:
            if suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
                # Stream order varies (ssm_alpha can precede attn_qkv); flush
                # whenever all four parts are present.
                part = {
                    "attn_qkv.weight": "qkv",
                    "attn_gate.weight": "z",
                    "ssm_beta.weight": "b",
                    "ssm_alpha.weight": "a",
                }[suffix]
                t_d = _dense_tensor(t, sides.get(name, {}))
                if suffix == "attn_qkv.weight":
                    # rows [q | k | v]: permute the v segment's head blocks
                    kd = _lg.key_head_dim * _lg.num_key_heads
                    t_d = torch.cat([t_d[: 2 * kd], _perm_v(t_d[2 * kd:])], dim=0)
                else:
                    t_d = _perm_v(t_d)
                in_proj_buf.setdefault(layer, {})[part] = t_d
                slots = in_proj_buf.get(layer)
                if all(k in slots for k in ("qkv", "z", "b", "a")):
                    yield from flush_in_proj(layer, base)
            elif suffix == "ssm_a":
                # The GGUF stores the decay rate -exp(A_log) (negative);
                # the model keeps A_log (log-space, exempt from the dtype
                # downcast): A_log = log(-rate).
                rate = _to_f32(t)
                yield f"{base}.linear_attn.A_log", torch.log(-_perm_v(rate))
            elif suffix == "ssm_dt.bias":
                yield f"{base}.linear_attn.dt_bias", _perm_v(_to_f32(t))
            elif suffix == "ssm_norm.weight":
                # GDN gated norm: a standard weight*x RMS norm (NOT Gemma +1).
                yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
            elif suffix == "ssm_out.weight":
                yield f"{base}.linear_attn.out_proj.weight", _perm_v(
                    _dense_tensor(t, sides.get(name, {})), dim=1)
            elif suffix == "ssm_conv1d.weight":
                # [conv_dim, K] -> the model's [conv_dim, 1, K].
                conv_w = _to_bf16(t)
                kd = _lg.key_head_dim * _lg.num_key_heads
                yield f"{base}.linear_attn.conv1d.weight", torch.cat(
                    [conv_w[: 2 * kd], _perm_v(conv_w[2 * kd:])], dim=0
                ).unsqueeze(1)
            elif suffix.startswith("ffn_") or suffix.startswith(("attn_norm", "post_attention")):
                pass  # handled by the shared MoE dispatch below
            else:
                raise ValueError(f"unmapped qwen35moe linear-attn tensor: {name}")

        # Both layer kinds carry the MoE block.
        if suffix == "ffn_gate_inp.weight":
            gate = _to_bf16(t).reshape(config.num_experts, config.hidden_size)
            yield f"{base}.mlp.gate.weight", gate
        elif suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, config.hidden_size)
        elif suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            shexp_buf.setdefault(layer, {})["gate" if suffix.startswith("ffn_gate") else "up"] = _dense_tensor(t, sides.get(name, {}))
            slots = shexp_buf.get(layer)
            if "gate" in slots and "up" in slots:
                yield from flush_shexp(layer, base)
        elif suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.weight", _dense_tensor(t, sides.get(name, {}))

    assert not in_proj_buf, f"incomplete in_proj groups: {sorted(in_proj_buf)}"
    assert not shexp_buf, f"incomplete shared-expert groups: {sorted(shexp_buf)}"
    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "weight_format", None) == "gguf"


# --------------------------------------------------------------------------------------
# Routed-expert host banks: GGML NVFP4 -> the offload cache's native "nvfp4"
# layout. Per expert and role, the GGUF stores [E, rows, H] packed with 4
# ue4m3 scale bytes + 32 nibble bytes per 64 elements, plus a per-expert
# fp32 global side tensor (``ffn_<role>_exps.scale``). The engine banks want:
#   gate_up_packed [rows, H//2] nibble bytes (lo = element 2j, hi = 2j+1),
#   gate_up_scale  [rows, H//16] fp8-e4m3 scale bytes,
#   gate_up_global [rows] fp16,
#   down_* likewise over [H, I].
# The ue4m3 bytes are reused verbatim as the engine's fp8-e4m3 scales (the
# ggml 0.5 factor and doubled kvalues cancel against the engine's decode), so
# the conversion is a byte permutation + the per-row global expand.
# --------------------------------------------------------------------------------------


def _nvfp4_ggml_to_engine_rows(raw: torch.Tensor, rows_per_expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One ggml NVFP4 tensor ``[E, rows_per_expert, row_bytes]`` -> engine
    ``(packed [E, rows, cols//2] uint8, scale [E, rows, cols//16] uint8)``.

    The two layouts differ in element order, not scale semantics:
    ggml nibble byte ``o`` of super-block ``s`` holds sub-block ``r = o//8``'s
    elements ``16r + (o%8)`` (lo) and ``16r + 8 + (o%8)`` (hi) -- non-consecutive
    pairs; the engine byte ``j`` holds the consecutive pair ``(2j, 2j+1)`` with
    block ``b = j//8``. Scales carry over 1:1: engine block ``b`` = ggml
    sub-block ``(s, r) = (b//4, b%4)``; ue4m3 bytes decode identically as the
    engine's positive fp8-e4m3 scales.
    """
    expert_total, rows_per_expert, row_bytes = raw.shape
    sb_per_row = row_bytes // _NVFP4_BYTES
    cols = sb_per_row * _NVFP4_BLOCK
    blocks = raw.reshape(expert_total, rows_per_expert, sb_per_row, _NVFP4_BYTES)
    scales = blocks[..., :4]   # [E, rows, sb, 4]
    nibbles = blocks[..., 4:]  # [E, rows, sb, 32]

    # ggml element stream: code(e) where e = 16r+i (lo of byte 8r+i) and
    # e = 16r+8+i (hi of byte 8r+i), r in 0..3, i in 0..7.
    lo = nibbles & 0xF  # [E, rows, sb, 32]
    hi = nibbles >> 4
    lo_m = lo.view(*lo.shape[:-1], 4, 8)   # [..., r, i]
    hi_m = hi.view(*hi.shape[:-1], 4, 8)
    codes = torch.empty(
        (*nibbles.shape[:-1], 4, 16), dtype=torch.uint8
    )  # [..., r, j] j in 0..15: j<8 -> lo[:, r, j], else hi[:, r, j-8]
    codes[..., :, :8] = lo_m
    codes[..., :, 8:] = hi_m
    codes = codes.reshape(*nibbles.shape[:-1], 64)  # [..., sb, 64] element order

    # Repack into consecutive engine pairs: byte j = code(2j) | code(2j+1) << 4.
    codes = codes.reshape(expert_total, rows_per_expert, sb_per_row, 32, 2)
    packed = codes[..., 0] | (codes[..., 1].to(torch.int32) << 4).to(torch.uint8)
    packed = packed.reshape(expert_total, rows_per_expert, cols // 2)

    # engine block b (16 elems) = ggml sub-block (s=b//4, r=b%4).
    scale = scales.reshape(expert_total, rows_per_expert, sb_per_row * 4)
    return packed, scale.reshape(expert_total, rows_per_expert, cols // 16)


def _q4_role_globals(model_path: str, role: str) -> torch.Tensor:
    """Per-expert fp32 global scales for a routed-expert role from its ``.scale``."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    suffix = f"ffn_{role}_exps.scale"
    for t in iter_gguf_tensors(model_path):
        if t.name.endswith(suffix):
            return _side_scale(t)
    raise KeyError(f"missing expert global scale tensor *.{suffix}")


def _expert_bank_specs(config: ModelConfig) -> dict[str, tuple[tuple[int, ...], int]]:
    """(shape, row_bytes) per role for the GGML NVFP4 expert tensors."""
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    return {
        "gate": ((E, I, H), H // _NVFP4_BLOCK * _NVFP4_BYTES),
        "up": ((E, I, H), H // _NVFP4_BLOCK * _NVFP4_BYTES),
        "down": ((E, H, I), I // _NVFP4_BLOCK * _NVFP4_BYTES),
    }


def load_nvfp4_expert_sources(
    model_path: str, config: ModelConfig, *, dummy: bool = False
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts in the offload cache's native
    ``"nvfp4"`` schema: per layer, ``gate_up_packed [E, 2I, H//2]``,
    ``gate_up_scale [E, 2I, H//16]``, ``gate_up_global [E, 2I]`` fp16,
    ``down_packed [E, H, I//2]``, ``down_scale [E, H, I//16]``,
    ``down_global [E, H]`` fp16."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    _require_tp1("expert banks")
    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    L = config.num_moe_layers

    if dummy:
        # Random packed bytes; scales mid-range so no decode under/overflows.
        gen = torch.Generator().manual_seed(0)
        return {
            "gate_up_packed": [torch.randint(0, 256, (E, 2 * I, H // 2), generator=gen, dtype=torch.uint8) for _ in range(L)],
            "gate_up_scale": [torch.randint(0x08, 0x78, (E, 2 * I, H // 16), generator=gen, dtype=torch.uint8) for _ in range(L)],
            "gate_up_global": [torch.ones(E, 2 * I, dtype=torch.float16) for _ in range(L)],
            "down_packed": [torch.randint(0, 256, (E, H, I // 2), generator=gen, dtype=torch.uint8) for _ in range(L)],
            "down_scale": [torch.randint(0x08, 0x78, (E, H, I // 16), generator=gen, dtype=torch.uint8) for _ in range(L)],
            "down_global": [torch.ones(E, H, dtype=torch.float16) for _ in range(L)],
        }

    gate_g = _q4_role_globals(model_path, "gate")
    up_g = _q4_role_globals(model_path, "up")
    down_g = _q4_role_globals(model_path, "down")

    banks: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("gate_up_packed", "gate_up_scale", "gate_up_global",
                        "down_packed", "down_scale", "down_global")
    }

    layer_of = lambda name: int(name.split(".")[1])

    def bank_globals(global_scale: torch.Tensor, rows: int) -> torch.Tensor:
        """Per-expert fp32 globals ``[n]`` -> ``[E, rows]`` fp16: expert ``e``'s
        every row carries ``global_scale[e]`` (``n == E``; ``rows`` is per-
        expert)."""
        assert global_scale.numel() == E
        return global_scale.to(torch.float16).reshape(E, 1).expand(E, rows).contiguous()

    def gate_up_bank_globals() -> torch.Tensor:
        """gate_up globals: the fused bank's row split is [gate I, up I] per
        expert, and gate/up carry distinct per-expert globals."""
        gate_half = gate_g.to(torch.float16).reshape(E, 1, 1).expand(E, I, 1)
        up_half = up_g.to(torch.float16).reshape(E, 1, 1).expand(E, I, 1)
        return torch.cat([gate_half, up_half], dim=1).reshape(E, 2 * I)

    packed_by_role: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for t in iter_gguf_tensors(model_path):
        suffix = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        if not any(suffix.startswith(p) for p in _EXPERT_SKIP_PREFIXES):
            continue
        if not suffix.endswith(".weight"):
            continue
        layer = layer_of(t.name)
        role = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}[
            suffix[: -len(".weight")]
        ]
        expected_shape, expected_row_bytes = _expert_bank_specs(config)[role]
        assert tuple(t.shape) == expected_shape, f"{t.name}: {tuple(t.shape)} != {expected_shape}"
        assert t.row_bytes == expected_row_bytes
        packed, scale = _nvfp4_ggml_to_engine_rows(
            t.packed().clone().reshape(*expected_shape[:2], expected_row_bytes),
            expected_shape[1],
        )
        packed_by_role[(layer, role)] = (packed, scale)
    # Allocate the per-layer banks INSIDE HostBank page-aligned buffers and
    # fill them there. Pinning torch-allocator pages (cudaHostRegister over
    # sub-ranges of one big allocation) wedged both driver versions we tried
    # ("already mapped" + silent process death), while HostBank's own mmap
    # buffers register cleanly.
    from freetoken.moe.host_banks import HostBank, pin_banks

    specs = {
        "gate_up_packed": ((E, 2 * I, H // 2), torch.uint8),
        "gate_up_scale": ((E, 2 * I, H // 16), torch.uint8),
        "gate_up_global": ((E, 2 * I), torch.float16),
        "down_packed": ((E, H, I // 2), torch.uint8),
        "down_scale": ((E, H, I // 16), torch.uint8),
        "down_global": ((E, H), torch.float16),
    }
    hb = {name: [HostBank(shape, dtype) for _ in range(L)] for name, (shape, dtype) in specs.items()}

    for layer in range(L):
        (gp, gs) = packed_by_role.pop((layer, "gate"))
        (upp, ups) = packed_by_role.pop((layer, "up"))
        (dp, ds) = packed_by_role.pop((layer, "down"))
        # gate rows first, then up rows: the gate_up GEMM's split order.
        hb["gate_up_packed"][layer].tensor.copy_(
            torch.cat([gp, upp], dim=1).reshape(E, 2 * I, H // 2)
        )
        hb["gate_up_scale"][layer].tensor.copy_(
            torch.cat([gs, ups], dim=1).reshape(E, 2 * I, H // 16)
        )
        hb["gate_up_global"][layer].tensor.copy_(gate_up_bank_globals().reshape(E, 2 * I))
        hb["down_packed"][layer].tensor.copy_(dp.reshape(E, H, I // 2))
        hb["down_scale"][layer].tensor.copy_(ds.reshape(E, H, I // 16))
        hb["down_global"][layer].tensor.copy_(bank_globals(down_g, H).reshape(E, H))
        for name in specs:
            banks[name].append(hb[name][layer].tensor)

    import os as _os
    if _os.environ.get("FREETOKEN_ZERO_EXPERTS"):
        for k in ("gate_up_global", "down_global"):
            for i in range(len(banks[k])):
                banks[k][i].zero_()
        print("FREETOKEN_ZERO_EXPERTS: expert globals zeroed", flush=True)

    # Settle every filled bank. cudaHostRegister at this scale (18 GiB)
    # kills the worker on both driver versions we tried (silent death, no
    # kernel taint — driver-level fault in the register path), so settle to
    # OS-locked residency instead: mlock keeps the banks resident (no swap
    # thrash — the actual failure we saw) without driver registration.
    # The offload cache's copy paths fall back to pageable copies for
    # non-registered banks.
    from freetoken.moe.host_banks import HostResidency

    for layer_banks in hb.values():
        for bank in layer_banks:
            bank.lock()
    assert not packed_by_role, f"missing expert tensors: {sorted(packed_by_role)}"
    return banks


def dummy_nvfp4_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    return load_nvfp4_expert_sources("", config, dummy=True)


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "is_gguf_model",
    "load_nvfp4_expert_sources",
    "dummy_nvfp4_expert_sources",
]