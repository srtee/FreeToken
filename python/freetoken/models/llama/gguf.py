"""Llama GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata
and map GGUF tensors onto the llama modules.

Dense model: no routed experts. The projections and the embedding stay in their
native packed block layout (``.qweight``); norms dequantize to bf16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _require_tp1(what: str) -> None:
    """GGUF quant layers are not sharded; reject TP>1 with a clear error."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size != 1:
        raise NotImplementedError(
            f"llama GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers are not tensor-parallel sharded)."
        )


def _tensor_types(model_path: str) -> dict:
    """Single-pass over GGUF tensors: quant type, bias, layer index."""
    from freetoken.models.gguf.reader import iter_gguf_tensors, tensor_type_of

    types: dict = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            types["token_embd"] = t.ggml_type
        elif name == "output_norm.weight":
            types["output_norm"] = t.ggml_type
        elif name == "output.weight":
            types["output"] = t.ggml_type
        elif name.endswith(".bias"):
            types["qkv_bias"] = True

        if name.startswith("blk."):
            parts = name.split(".")
            layer = int(parts[1])
            suffix = ".".join(parts[2:])

            if suffix == "attn_q.weight":
                types.setdefault("attn_q", {})
                types["attn_q"][layer] = t.ggml_type
            elif suffix == "attn_k.weight":
                types.setdefault("attn_k", {})
                types["attn_k"][layer] = t.ggml_type
            elif suffix == "attn_v.weight":
                types.setdefault("attn_v", {})
                types["attn_v"][layer] = t.ggml_type
            elif suffix == "attn_output.weight":
                types.setdefault("attn_output", {})
                types["attn_output"][layer] = t.ggml_type
            elif suffix == "ffn_gate.weight":
                types.setdefault("ffn_gate", {})
                types["ffn_gate"][layer] = t.ggml_type
            elif suffix == "ffn_up.weight":
                types.setdefault("ffn_up", {})
                types["ffn_up"][layer] = t.ggml_type
            elif suffix == "ffn_down.weight":
                types.setdefault("ffn_down", {})
                types["ffn_down"][layer] = t.ggml_type

    types.setdefault("qkv_bias", False)
    return types


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata
    arch = shim.model_type

    def g(key: str):
        val = m.get(f"{arch}.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key {arch}.{key}")
        return val

    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    num_kv_heads = int(g("attention.head_count_kv"))
    head_dim = hidden // num_qo_heads
    max_pos = int(g("context_length"))

    return ModelConfig(
        num_layers=int(g("block_count")),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=int(shim.vocab_size),
        intermediate_size=int(g("feed_forward_length")),
        hidden_act="silu",
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=head_dim,
            max_position=max_pos,
            base=float(g("rope.freq_base")),
            # llama.cpp carries no rope-scaling extension here (linear/yarn would appear
            # as rope.scaling.* KV); plain default rope.
            scaling=None,
        ),
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        model_type=arch,
        architectures=list(shim.architectures),
        weight_format="gguf",
        gguf_type_table=_tensor_types(shim.model_path),
    )


# Per-layer norm tensors (gguf suffix -> freetoken module-relative name).
_LAYER_NORM_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


def _to_bf16(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every llama param from a GGUF file.

    Quantized projections keep their packed block layout (``.qweight``). q/k/v and
    gate/up fuse by concatenating packed rows along the output dim when the shards
    share one ggml type; mixed-type checkpoints yield the qkv shards separately
    for a ``GGUFSplitQKV``.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    from freetoken.utils import cached_load_hf_config

    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}

    def layer_of(name: str) -> int:
        return int(name.split(".")[1])

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _to_bf16(t)
            continue
        if not name.startswith("blk."):
            continue

        layer = layer_of(name)
        suffix = name.split(".", 2)[2]
        base = f"model.layers.{layer}"

        if suffix in _LAYER_NORM_MAP:
            yield f"{base}.{_LAYER_NORM_MAP[suffix]}", _to_bf16(t)
            continue

        if suffix == "attn_q.weight":
            qkv_buf.setdefault(layer, {})["q"] = t.packed()
        elif suffix == "attn_k.weight":
            qkv_buf.setdefault(layer, {})["k"] = t.packed()
        elif suffix == "attn_v.weight":
            qkv_buf.setdefault(layer, {})["v"] = t.packed()
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        elif suffix == "ffn_gate.weight":
            gate_up_buf.setdefault(layer, {})["gate"] = t.packed()
        elif suffix == "ffn_up.weight":
            gate_up_buf.setdefault(layer, {})["up"] = t.packed()
        elif suffix == "ffn_down.weight":
            yield f"{base}.mlp.down_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped llama GGUF tensor: {name}")

        slots = qkv_buf.get(layer)
        if slots is not None and all(k in slots for k in ("q", "k", "v")):
            types = config.gguf_type_table
            qt, kt, vt = types["attn_q"][layer], types["attn_k"][layer], types["attn_v"][layer]
            if qt == kt == vt:
                # Uniform type: packed rows share row_bytes, concat into the fused tensor.
                yield f"{base}.self_attn.qkv_proj.qweight", torch.cat(
                    [slots["q"], slots["k"], slots["v"]], dim=0
                )
                del qkv_buf[layer]
            else:
                # Mixed types: packed rows have different widths, keep separate shards.
                yield f"{base}.self_attn.qkv_proj.q_proj.qweight", slots["q"]
                yield f"{base}.self_attn.qkv_proj.k_proj.qweight", slots["k"]
                yield f"{base}.self_attn.qkv_proj.v_proj.qweight", slots["v"]
                del qkv_buf[layer]
        gu = gate_up_buf.get(layer)
        if gu is not None and all(k in gu for k in ("gate", "up")):
            yield f"{base}.mlp.gate_up_proj.qweight", torch.cat(
                [gu["gate"], gu["up"]], dim=0
            )
            del gate_up_buf[layer]

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not gate_up_buf, f"incomplete gate_up groups: {sorted(gate_up_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "weight_format", None) == "gguf"


def convert_llama_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace llama's dense projections + embedding with native GGUF ops.

    Types come from the per-tensor table parse_gguf_config stashed in
    ``config.gguf_type_table`` (Q3_K_M-style mixed quants quantize tensors individually).
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFSplitQKV
    from freetoken.layers.gguf import GGUFTiedLMHead, GGUFUntiedLMHead

    types = config.gguf_type_table

    embed_type = types["token_embd"]
    model.model.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=embed_type,
    )
    for layer in model.model.layers.op_list:
        lid = layer._layer_id
        attn = layer.self_attn
        qt, kt, vt = types["attn_q"][lid], types["attn_k"][lid], types["attn_v"][lid]
        if qt == kt == vt:
            out_features, in_features = attn.qkv_proj.weight.shape
            attn.qkv_proj = GGUFLinear(
                in_features, out_features, qt, has_bias=False,
            )
        else:
            attn.qkv_proj = GGUFSplitQKV(
                config.hidden_size,
                attn.qo_attn_dim,
                attn.kv_attn_dim,
                qt, kt, vt,
                has_bias=False,
            )
        for owner, attr, type_key in (
            (layer.self_attn, "o_proj", "attn_output"),
            (layer.mlp, "gate_up_proj", "ffn_gate"),
            (layer.mlp, "down_proj", "ffn_down"),
        ):
            lin = getattr(owner, attr)
            out_features, in_features = lin.weight.shape
            has_bias = lin.bias is not None
            qtype = types[type_key][lid]
            setattr(
                owner,
                attr,
                GGUFLinear(
                    in_features,
                    out_features,
                    qtype,
                    has_bias=has_bias,
                ),
            )
    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(model.model.embed_tokens, embed_type)
    else:
        model.lm_head = GGUFUntiedLMHead(
            config.hidden_size,
            config.vocab_size,
            types["output"],
        )


__all__ = ["parse_gguf_config", "iter_gguf_weights", "is_gguf_model", "convert_llama_to_gguf"]
