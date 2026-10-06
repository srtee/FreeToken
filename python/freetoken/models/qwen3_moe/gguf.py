"""Qwen3-MoE GGUF adapter: build the FreeToken ``ModelConfig`` from GGUF metadata
and map GGUF tensors onto the qwen3_moe modules.

Mirrors ``models/qwen2/gguf.py`` (same attention geometry, per-tensor quant
types) and ``models/gemma4/gguf.py`` (MoE expert routing). The dense params and
the embedding stay in their native packed block layout (``.qweight``); norms,
router gate and biases dequantize. The routed experts (``ffn_gate_exps`` /
``ffn_up_exps`` / ``ffn_down_exps``) and the router projection
(``ffn_gate_inp``) are handled by the offload-bank loader (part B): for now
``parse_gguf_config`` routes them through ``expert_quant="q4_0"`` /
``moe_weight_format="q4_0"`` like gemma4's GGUF path, and ``iter_gguf_weights``
skips them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import ModelConfig, RotaryConfig
from freetoken.models.gguf.dequant import dequantize

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 loudly."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size != 1:
        raise NotImplementedError(
            f"qwen3moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


def _tensor_types(model_path: str) -> dict:
    """Single pass over GGUF tensors: quant type of every mapped tensor + bias presence.

    The qkv fusion needs the per-tensor types (Q3_K_M-style checkpoints quantize
    q/k and v differently); the swap helper reads the same table.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

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
        elif name.startswith("blk."):
            parts = name.split(".")
            layer = int(parts[1])
            suffix = ".".join(parts[2:])
            if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                          "attn_output.weight"):
                key = suffix.split(".")[0]
                types.setdefault(key, {})
                types[key][layer] = t.ggml_type
    return types


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    m = shim.metadata
    arch = shim.model_type  # "qwen3moe"

    def g(key: str):
        val = m.get(f"{arch}.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key {arch}.{key}")
        return val

    hidden = int(g("embedding_length"))
    num_qo_heads = int(g("attention.head_count"))
    # qwen3 GGUF writes a per-layer kv-head list (one entry per layer); every
    # layer shares the value, like the HF scalar num_key_value_heads.
    kv_heads = g("attention.head_count_kv")
    num_kv_heads = int(kv_heads[0] if isinstance(kv_heads, (list, tuple)) else kv_heads)
    head_dim = int(g("attention.key_length"))
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
            scaling=None,
        ),
        num_experts=int(g("expert_count")),
        num_experts_per_tok=int(g("expert_used_count")),
        moe_intermediate_size=int(g("expert_feed_forward_length")),
        norm_topk_prob=True,
        model_type=arch,
        architectures=list(shim.architectures),
        moe_enabled=True,
        # Routed experts route through the native-Q4_0 offload-bank path, like
        # the gemma4 GGUF path. Part B owns the three-tensor bank loader
        # (ffn_gate_exps / ffn_up_exps / ffn_down_exps).
        expert_quant="q4_0",
        moe_weight_format="q4_0",
        weight_format="gguf",
        use_qk_norm=True,
        gguf_type_table=_tensor_types(shim.model_path),
    )


# Per-layer norm tensors (gguf suffix -> freetoken module-relative name).
_LAYER_NORM_MAP = {
    # llama.cpp's qwen3 conversion: attn_norm -> HF input_layernorm,
    # ffn_norm -> HF post_attention_layernorm (qwen3 has exactly these two).
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    # qwen3's QK-norm weights (has_qk_norm=True -> self_attn.q_norm/k_norm).
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

# Routed-expert weight tensors: handled by the offload bank loader, skipped
# here. The router projection (ffn_gate_inp) is NOT skipped — it maps onto
# mlp.gate like a dense param.
_EXPERT_SKIP_SUFFIXES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def _to_bf16(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16)
    return flat.reshape(t.shape)


def _to_f32(t) -> torch.Tensor:
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.float32)
    return flat.reshape(t.shape)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen3moe param.

    Quantized projections keep their packed block layout (``.qweight``). q/k/v
    fuse into ``qkv_proj.qweight`` by concatenating packed rows along the output
    dim when the shards share one ggml type; mixed-type checkpoints yield the
    qkv shards separately for a ``GGUFSplitQKV``. Attention biases (qwen3 has
    QK-norm but real q/k biases) dequantize to f32. Routed experts and the
    router projection are skipped (part B owns them).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    from freetoken.utils import cached_load_hf_config

    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    bias_buf: dict[int, dict[str, torch.Tensor]] = {}

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
        if any(name.endswith(sfx) for sfx in _EXPERT_SKIP_SUFFIXES):
            continue  # routed experts + router -> offload banks (part B)

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
        elif suffix == "attn_q.bias":
            bias_buf.setdefault(layer, {})["q"] = _to_f32(t)
        elif suffix == "attn_k.bias":
            bias_buf.setdefault(layer, {})["k"] = _to_f32(t)
        elif suffix == "attn_v.bias":
            bias_buf.setdefault(layer, {})["v"] = _to_f32(t)
        elif suffix == "ffn_gate_inp.weight":
            gate = _to_bf16(t).reshape(config.num_experts, config.hidden_size)
            yield f"{base}.mlp.gate.weight", gate
        elif suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
        else:
            raise ValueError(f"unmapped qwen3moe GGUF tensor: {name}")

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
        bb = bias_buf.get(layer)
        # qwen3 attention biases only q and k (v is bias-free, matching the HF
        # Qwen3Moe layout: q_proj/k_proj bias=True, v_proj bias=False). The
        # fused qkv_proj bias row layout is q rows, k rows, then v rows, so for
        # the fused path pad the v segment with zeros; for the split path yield
        # the two real shards.
        if bb is not None and "q" in bb and "k" in bb:
            types = config.gguf_type_table
            qt, kt, vt = types["attn_q"][layer], types["attn_k"][layer], types["attn_v"][layer]
            v_rows = bb["k"].shape[0]
            if qt == kt == vt:
                v_pad = torch.zeros(v_rows, dtype=bb["k"].dtype)
                yield f"{base}.self_attn.qkv_proj.bias", torch.cat(
                    [bb["q"], bb["k"], v_pad], dim=0
                )
            else:
                yield f"{base}.self_attn.qkv_proj.q_proj.bias", bb["q"]
                yield f"{base}.self_attn.qkv_proj.k_proj.bias", bb["k"]
            del bias_buf[layer]

    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not bias_buf, f"incomplete qkv bias groups: {sorted(bias_buf)}"


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return getattr(config, "weight_format", None) == "gguf"


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native Q4_0, three-tensor layout) for the offload cache.
#
# llama.cpp's qwen3moe conversion stores gate/up/down as SEPARATE 3D tensors
# (ffn_gate_exps / ffn_up_exps / ffn_down_exps, [E, rows, H] ggml shape), unlike
# gemma4's fused ffn_gate_up_exps. The offload cache's "q4_0" schema wants a
# fused gate_up [E, 2I, row_bytes(H)] bank, so gate and up are concatenated along
# the output-row dim at load (packed rows over the same input dim concat cleanly,
# exactly like the fused qkv path in iter_gguf_weights).
# --------------------------------------------------------------------------------------


def _q4_0_expert_specs(config: ModelConfig) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes

    return {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_Q4_0)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_Q4_0)), torch.uint8),
    }


def _q4_0_to_q4_1(packed: torch.Tensor, n_blocks_per_row: int) -> torch.Tensor:
    """Upconvert Q4_0 rows to the Q4_1 block layout, bit-exact.

    Q4_0 block: [d:fp16][16B nibbles] (18B/32 values), decodes ``d*(x-8)``;
    Q4_1: [d:fp16][m:fp16][16B nibbles] (20B), decodes ``d*x + m``. Writing
    ``m = fp16(-8*d)`` with the same d and nibbles reproduces Q4_0's values
    exactly (the kernel computes both sides in fp32, and fp16(-8*d) is exact
    for every fp16 d — only the exponent shifts).
    """
    rows = packed.shape[0]
    blocks = packed.reshape(rows, n_blocks_per_row, 18)
    d = blocks[:, :, 0:2]
    nib = blocks[:, :, 2:18]
    d_f16 = d.reshape(-1, 2).view(torch.float16).to(torch.float32)
    m_f16 = (-8.0 * d_f16).to(torch.float16).view(torch.uint8).reshape(
        rows, n_blocks_per_row, 2
    )
    out = torch.zeros(rows, n_blocks_per_row, 20, dtype=torch.uint8)
    out[:, :, 0:2] = d
    out[:, :, 2:4] = m_f16
    out[:, :, 4:20] = nib
    return out.reshape(rows, n_blocks_per_row * 20)


def load_q4_0_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> tuple[dict[str, list[torch.Tensor]], dict[str, int]]:
    """Per-layer host banks of the routed experts' packed block bytes + their ggml types.

    Same contract as gemma4's ``load_q4_0_expert_sources`` (gate_up one
    ``[E, 2I, row_bytes(H)]`` tensor per layer, down one ``[E, H,
    row_bytes(I)]``, streamed through ``layer_sink`` / pin-after-fill), extended
    for the three-tensor layout: gate and up land in the fused bank's first and
    second ``I`` row halves, down keeps its own bank. The ggml quant type is
    probed per role from the tensor table; the DOWN projection normalizes to
    Q4_1 (qwen3moe checkpoints mix Q4_1 and Q4_0 down layers — llama.cpp
    quantizes the first-few layers' down tensors Q4_1, the rest Q4_0 — and
    Q4_0 -> Q4_1 with m=0 is bit-exact, so one bank layout serves both). The
    types map feeds ``ExpertBanks.ggml_types`` -> the fused MoE kernel's
    per-role dispatch.
    """
    from freetoken.models.gguf.dequant import GGML_Q4_0, GGML_Q4_1, row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size

    # Pass 1: cheap metadata probe — the ggml type of each expert role. gate/up
    # must be uniform; down may mix Q4_0/Q4_1 (normalized to Q4_1 below).
    types: dict[str, set] = {"gate_up": set(), "down": set()}
    for t in iter_gguf_tensors(model_path):
        if not t.name.startswith("blk."):
            continue
        if t.name.endswith("ffn_gate_exps.weight") or t.name.endswith("ffn_up_exps.weight"):
            types["gate_up"].add(t.ggml_type)
        elif t.name.endswith("ffn_down_exps.weight"):
            types["down"].add(t.ggml_type)
    assert len(types["gate_up"]) == 1, f"mixed gate/up expert types: {types['gate_up']}"
    assert types["gate_up"] <= {GGML_Q4_0}, f"unsupported gate/up type: {types['gate_up']}"
    assert types["down"] <= {GGML_Q4_0, GGML_Q4_1}, f"unsupported down types: {types['down']}"
    gt = next(iter(types["gate_up"]))
    dt = GGML_Q4_1  # normalized bank type

    specs = {
        "gate_up": ((E, 2 * I, row_bytes(H, gt)), torch.uint8),
        "down": ((E, H, row_bytes(I, dt)), torch.uint8),
    }
    hb = alloc_layer_banks(specs, L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen_g, seen_u, seen_dn = set(), set(), set()

    def _load(sink) -> None:
        tracker = (
            LayerCompletionTracker(3, hb, sink) if sink is not None else None
        )  # gate + up + down
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if t.name.endswith("ffn_gate_exps.weight"):
                banks["gate_up"][layer][:, :I].copy_(
                    t.packed().reshape(E, I, row_bytes(H, gt))
                )
                seen_g.add(layer)
            elif t.name.endswith("ffn_up_exps.weight"):
                banks["gate_up"][layer][:, I:].copy_(
                    t.packed().reshape(E, I, row_bytes(H, gt))
                )
                seen_u.add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                packed = t.packed()
                if t.ggml_type == GGML_Q4_0:
                    packed = _q4_0_to_q4_1(
                        packed.reshape(E * H, row_bytes(I, GGML_Q4_0)),
                        I // 32,
                    )
                banks["down"][layer].copy_(packed.reshape(E, H, row_bytes(I, dt)))
                seen_dn.add(layer)
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)

    want = set(range(L))
    assert seen_g == want and seen_u == want and seen_dn == want, (
        f"missing expert layers: gate {sorted(want - seen_g)}, "
        f"up {sorted(want - seen_u)}, down {sorted(want - seen_dn)}"
    )
    return banks, {"gate_up": gt, "down": dt}


def dummy_q4_0_expert_sources(config: ModelConfig) -> dict[str, list[torch.Tensor]]:
    """Random packed expert banks shaped like ``load_q4_0_expert_sources`` output
    (down normalized to Q4_1, matching the loader's bank layout)."""
    from freetoken.models.gguf.dequant import GGML_Q4_1, row_bytes
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    E = config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    specs = {
        "gate_up": ((E, 2 * I, row_bytes(H, GGML_Q4_0)), torch.uint8),
        "down": ((E, H, row_bytes(I, GGML_Q4_1)), torch.uint8),
    }
    hb = alloc_layer_banks(specs, config.num_layers)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill
    return banks


def convert_qwen3moe_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace qwen3-moe's dense projections + embedding with native GGUF ops.

    Same swap as qwen2's (attention qkv + o_proj + embedding + lm_head); the MoE
    layer needs no swap (its experts come from the q4_0 offload banks, its gate
    stays a dense LinearReplicated fed by ``mlp.gate.weight``). Types come from
    ``config.gguf_type_table`` (mixed-quant checkpoints quantize q/k and v
    individually -> GGUFSplitQKV).
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
        has_bias = attn.qkv_proj.bias is not None
        if qt == kt == vt:
            attn.qkv_proj = GGUFLinear(
                config.hidden_size,
                attn.qkv_proj.weight.shape[0],
                qt,
                has_bias=has_bias,
            )
        else:
            attn.qkv_proj = GGUFSplitQKV(
                config.hidden_size,
                attn.qo_attn_dim,
                attn.kv_attn_dim,
                qt,
                kt,
                vt,
                has_bias=has_bias,
            )
        out_features, in_features = attn.o_proj.weight.shape
        attn.o_proj = GGUFLinear(
            in_features, out_features, types["attn_output"][lid], has_bias=False
        )
    if config.tie_word_embeddings:
        model.lm_head = GGUFTiedLMHead(model.model.embed_tokens, embed_type)
    else:
        model.lm_head = GGUFUntiedLMHead(
            config.hidden_size,
            config.vocab_size,
            types["output"],
        )


__all__ = [
    "parse_gguf_config",
    "iter_gguf_weights",
    "convert_qwen3moe_to_gguf",
    "is_gguf_model",
    "load_q4_0_expert_sources",
    "dummy_q4_0_expert_sources",
]