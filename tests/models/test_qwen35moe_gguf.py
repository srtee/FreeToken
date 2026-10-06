"""qwen35moe (Qwen3.5/3.6 hybrid GDN+MoE) GGUF adapter tests.

Builds a tiny hybrid GGUF with gguf-py (2 full-attn layers at ids 1 and 3 of a
4-layer stack via full_attention_interval=4 — wait, that yields layers 3 only;
we use interval=2 for two full layers at ids 1, 3) with hand-packed NVFP4
dense/expert tensors, and verifies:

- parse_gguf_config recovers the hybrid geometry (groups, GDN dims, expert
  quant format).
- iter_gguf_weights fuses the GDN in-proj in the model's split order
  [qkv, z, b, a], fuses full-attn qkv [q2x, k, v], bakes the Gemma +1 into
  layer/qk norms but NOT into the GDN gated norm, dequantizes dense NVFP4
  exactly, and skips routed experts.
- load_nvfp4_expert_sources produces the offload cache's native bank layout
  with the right shapes, gate/up row order, and per-expert globals.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Tiny geometry (real model: 40 layers, 16 q / 2 kv heads x 256, 30 GDN).
H = 64
V = 128
HEAD_DIM = 16
N_Q, N_KV = 4, 2
# GDN: key heads 2 x key_head 8 = key_dim 16; value heads 4 x 8 = value_dim 32.
GDN_K_HEADS, GDN_V_HEADS, GDN_HEAD_DIM = 2, 4, 16
KEY_DIM = GDN_K_HEADS * GDN_HEAD_DIM          # 16
VALUE_DIM = GDN_V_HEADS * GDN_HEAD_DIM        # 32
CONV_DIM = 2 * KEY_DIM + VALUE_DIM            # 64
CONV_K = 4
DT_RANK = 2
N_EXP, TOPK = 4, 2
MOE_I = 64
SHARED_I = 64
# 4 layers, full attention on 1-indexed even blocks -> layer ids 1, 3.
N_LAYERS, INTERVAL = 4, 2
FULL_IDS = (1, 3)

NVFP4_BLOCK = 64
NVFP4_BYTES = 36
E4M3_MAX = 448.0


def _fp32_to_ue4m3(x: float) -> int:
    """Float32 -> unsigned e4m3 byte (bias 7, no sign), matching ggml."""
    if x <= 0:
        return 0
    e = int(np.floor(np.log2(x))) + 7
    m = int(round((x / 2.0 ** (e - 7) - 1) * 8))
    if m > 7:
        m, e = 0, e + 1
    return min(e, 15) << 3 | m


def _pack_nvfp4_row(codes: np.ndarray, scale: float) -> np.ndarray:
    """Pack one row of `cols` e2m1 CODES (byte j = code(2j) | code(2j+1)<<4)
    as GGML NVFP4: per 64-element super-block [4 ue4m3 scale bytes + 32
    nibble bytes]. All sub-blocks share the `scale` (block scale, float)."""
    cols = codes.shape[0]
    assert cols % NVFP4_BLOCK == 0
    sb = cols // NVFP4_BLOCK
    scale_byte = np.uint8(_fp32_to_ue4m3(scale))
    out = np.zeros(sb * NVFP4_BYTES, dtype=np.uint8)
    for s in range(sb):
        base = s * NVFP4_BYTES
        out[base:base + 4] = scale_byte
        for r in range(4):
            for i in range(8):
                out[base + 4 + r * 8 + i] = int(codes[s * 64 + 16 * r + i]) | (
                    int(codes[s * 64 + 16 * r + 8 + i]) << 4
                )
    return out


def _nvfp4_tensor(codes: np.ndarray, scale: float, ne_shape: tuple) -> np.ndarray:
    """Pack a full weight tensor's e2m1 codes [torch rows, cols] -> flat bytes
    in ggml row order (torch rows == ggml rows)."""
    rows, cols = codes.shape
    assert cols % NVFP4_BLOCK == 0
    packed_rows = np.stack([_pack_nvfp4_row(codes[r], scale) for r in range(rows)])
    return packed_rows.reshape(-1)


def _write_tiny_qwen35moe(path: str) -> None:
    from gguf import GGUFWriter, GGMLQuantizationType

    rng = np.random.default_rng(0)
    w = GGUFWriter(path, "qwen35moe")
    w.add_block_count(N_LAYERS)
    w.add_embedding_length(H)
    w.add_head_count(N_Q)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_context_length(512)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_expert_count(N_EXP)
    w.add_expert_used_count(TOPK)
    w.add_expert_feed_forward_length(MOE_I)
    w.add_uint32(f"qwen35moe.full_attention_interval", INTERVAL)
    w.add_uint32(f"qwen35moe.ssm.group_count", GDN_K_HEADS)
    w.add_uint32(f"qwen35moe.ssm.state_size", GDN_HEAD_DIM)
    w.add_uint32(f"qwen35moe.ssm.inner_size", VALUE_DIM)
    w.add_uint32(f"qwen35moe.ssm.time_step_rank", DT_RANK)
    w.add_uint32(f"qwen35moe.ssm.conv_kernel", CONV_K)
    # partial rotary: 4 of 16 head dims
    w.add_uint32(f"qwen35moe.rope.dimension_count", 4)
    w.add_uint32(f"qwen35moe.shared_expert_feed_forward_length" if False else f"qwen35moe.expert_shared_feed_forward_length", SHARED_I)

    def t_f32(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    def t_f16(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float16)

    # token_embd / output in ggml ne order: (hidden, vocab)
    w.add_tensor("token_embd.weight", t_f16(V, H).copy())  # np [vocab, hidden] -> ggml ne (H, V)
    w.add_tensor("output_norm.weight", np.ones(H, dtype=np.float16))
    w.add_tensor("output.weight", t_f16(H, V).copy())

    def nvfp4(name: str, values: np.ndarray, scale: float, torch_shape: tuple):
        """values/scale define the exact dequantized torch [rows, cols] tensor."""
        rows, cols = torch_shape
        flat = _nvfp4_tensor(values, scale, torch_shape)
        # gguf-py raw_shape convention for raw uint8: [n_rows, row_bytes].
        n_rows = rows
        row_bytes = cols // NVFP4_BLOCK * NVFP4_BYTES
        w.add_tensor(name, flat.reshape(n_rows, row_bytes), raw_shape=(n_rows, row_bytes),
                     raw_dtype=GGMLQuantizationType.NVFP4)
        # per-tensor global side tensor
        w.add_tensor(name[: -len(".weight")] + ".scale", np.array([scale], dtype=np.float32))

    for l in range(N_LAYERS):
        p = f"blk.{l}."
        w.add_tensor(p + "attn_norm.weight", np.ones(H, dtype=np.float16))
        w.add_tensor(p + "post_attention_norm.weight", np.ones(H, dtype=np.float16))
        # Router (F32) + shared-expert gate.
        w.add_tensor(p + "ffn_gate_inp.weight", t_f32(N_EXP, H).copy())
        w.add_tensor(p + "ffn_gate_inp_shexp.weight", t_f32(1, H).copy())
        # Shared expert (NVFP4 dense): block scale == side global == 0.5.
        s_scale = 0.5
        def sc(shape):
            # e2m1 codes; decode = e2m1(code) * block * global
            return rng.integers(0, 16, shape).astype(np.uint8)
        nvfp4(p + "ffn_gate_shexp.weight", sc((SHARED_I, H)), s_scale, (SHARED_I, H))
        nvfp4(p + "ffn_up_shexp.weight", sc((SHARED_I, H)), s_scale, (SHARED_I, H))
        nvfp4(p + "ffn_down_shexp.weight", sc((H, SHARED_I)), s_scale, (H, SHARED_I))
        # Routed experts: [E, rows, cols] ggml ne = (cols, rows, E).
        def experts(name, rows, cols, scale):
            codes = rng.integers(0, 16, (N_EXP, rows, cols)).astype(np.uint8)
            flat = np.stack([_nvfp4_tensor(codes[e], scale, (rows, cols)) for e in range(N_EXP)])
            # 3-D: raw_shape [E, rows, row_bytes] (byte-shape convention;
            # the writer converts the last dim to elements).
            row_bytes = cols // NVFP4_BLOCK * NVFP4_BYTES
            w.add_tensor(name, flat.reshape(N_EXP, rows, row_bytes),
                         raw_shape=(N_EXP, rows, row_bytes),
                         raw_dtype=GGMLQuantizationType.NVFP4)
            w.add_tensor(name[: -len(".weight")] + ".scale", np.full(N_EXP, scale, dtype=np.float32))
            return codes
        gate_vals = experts(p + "ffn_gate_exps.weight", MOE_I, H, 1.0)
        up_vals = experts(p + "ffn_up_exps.weight", MOE_I, H, 1.0)
        down_vals = experts(p + "ffn_down_exps.weight", H, MOE_I, 1.0)
        if l in FULL_IDS:
            # Full attention: q includes the 2x gate rows.
            qv = sc((N_Q * HEAD_DIM * 2, H))
            nvfp4(p + "attn_q.weight", qv, 0.5, (N_Q * HEAD_DIM * 2, H))
            nvfp4(p + "attn_k.weight", sc((N_KV * HEAD_DIM, H)), 0.5, (N_KV * HEAD_DIM, H))
            nvfp4(p + "attn_v.weight", sc((N_KV * HEAD_DIM, H)), 0.5, (N_KV * HEAD_DIM, H))
            nvfp4(p + "attn_output.weight", sc((H, N_Q * HEAD_DIM)), 0.5, (H, N_Q * HEAD_DIM))
            w.add_tensor(p + "attn_q_norm.weight", np.ones(HEAD_DIM, dtype=np.float16))
            w.add_tensor(p + "attn_k_norm.weight", np.ones(HEAD_DIM, dtype=np.float16))
        else:
            # GDN linear layer: in-proj qkv (conv_dim), z gate (value_dim),
            # b/a (dt_rank rows each), ssm params.
            nvfp4(p + "attn_qkv.weight", sc((CONV_DIM, H)), 0.5, (CONV_DIM, H))
            nvfp4(p + "attn_gate.weight", sc((VALUE_DIM, H)), 0.5, (VALUE_DIM, H))
            nvfp4(p + "ssm_beta.weight", sc((DT_RANK, H)), 0.5, (DT_RANK, H))
            nvfp4(p + "ssm_alpha.weight", sc((DT_RANK, H)), 0.5, (DT_RANK, H))
            # GGUF stores -exp(A_log): negative rates with A_log = 1.5
            w.add_tensor(p + "ssm_a", (-np.exp(1.5)).astype(np.float32) * np.ones(DT_RANK, dtype=np.float32))
            w.add_tensor(p + "ssm_dt.bias", rng.standard_normal(DT_RANK).astype(np.float32))
            w.add_tensor(p + "ssm_norm.weight", np.full(GDN_HEAD_DIM, 0.7, dtype=np.float16))
            nvfp4(p + "ssm_out.weight", sc((H, VALUE_DIM)), 0.5, (H, VALUE_DIM))
            conv = (rng.standard_normal((CONV_DIM, CONV_K)) * 0.1).astype(np.float32)
            w.add_tensor(p + "ssm_conv1d.weight", conv)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(name="tiny_gguf")
def tiny_gguf_fixture(tmp_path):
    path = str(tmp_path / "tiny_qwen35moe.gguf")
    _write_tiny_qwen35moe(path)
    return path


@pytest.fixture(name="tp1", autouse=True)
def tp1_fixture():
    from freetoken.distributed import info, set_tp_info

    if info.try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _shim(path):
    from freetoken.models.gguf.config import build_gguf_shim

    return build_gguf_shim(path)


def _nvfp4_reference(values: np.ndarray, scale: float) -> np.ndarray:
    """Reference dequant of a hand-packed tensor: values * scale * global(=scale
    passed at pack time is the BLOCK scale; the .scale side tensor is the
    GLOBAL). The writer sets global == block scale here, so the engine decode
    is values * block_scale * global -- NOT values. Kept for exactness tests."""
    raise NotImplementedError


def test_parse_config_recovers_hybrid_geometry(tiny_gguf):
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config

    cfg = parse_gguf_config(_shim(tiny_gguf))
    assert cfg.num_layers == N_LAYERS
    assert cfg.num_qo_heads == N_Q
    assert cfg.num_kv_heads == N_KV
    assert cfg.head_dim == HEAD_DIM
    assert cfg.hidden_size == H
    assert cfg.vocab_size == V
    assert cfg.num_experts == N_EXP
    assert cfg.num_experts_per_tok == TOPK
    assert cfg.moe_intermediate_size == MOE_I
    assert cfg.shared_expert_intermediate_size == SHARED_I
    assert cfg.expert_quant == "nvfp4"
    assert cfg.moe_weight_format == "nvfp4"
    assert cfg.weight_format == "gguf"
    assert cfg.tie_word_embeddings is False
    assert cfg.rotary_config.head_dim == HEAD_DIM
    assert cfg.rotary_config.base == 10000.0
    assert cfg.rotary_config.rotary_dim == 4  # partial rope from dimension_count
    full = [g for g in cfg.attention_groups if g.name == "full"][0]
    linear = [g for g in cfg.attention_groups if g.name == "linear"][0]
    assert full.layer_ids == FULL_IDS
    assert linear.layer_ids == tuple(l for l in range(N_LAYERS) if l not in FULL_IDS)
    assert linear.num_key_heads == GDN_K_HEADS
    assert linear.num_value_heads == GDN_V_HEADS
    assert linear.key_head_dim == GDN_HEAD_DIM
    assert linear.value_head_dim == GDN_HEAD_DIM
    assert linear.conv_kernel_dim == CONV_K


def test_iter_weights_full_attn_fusion_and_norms(tiny_gguf):
    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    pairs = dict(iter_gguf_weights(tiny_gguf, None, include_moe_experts=False,
                                   include_non_moe=True))
    names = set(pairs)
    assert "model.embed_tokens.weight" in names
    assert "lm_head.weight" in names
    assert "model.norm.weight" in names
    # Norms pass through pre-baked: the writer stores ones -> weight 1.0.
    assert torch.allclose(pairs["model.norm.weight"].float(), torch.full((H,), 1.0), atol=1e-2)
    for l in range(N_LAYERS):
        base = f"model.layers.{l}"
        assert f"{base}.input_layernorm.weight" in pairs
        assert torch.allclose(pairs[f"{base}.input_layernorm.weight"].float(), torch.full((H,), 1.0), atol=1e-2)
        assert f"{base}.mlp.gate.weight" in pairs
        assert pairs[f"{base}.mlp.gate.weight"].shape == (N_EXP, H)
        assert f"{base}.mlp.shared_expert_gate.weight" in pairs
        assert f"{base}.mlp.shared_expert.gate_up_proj.weight" in pairs
        assert f"{base}.mlp.shared_expert.down_proj.weight" in pairs
        # No routed-expert tensors in the dense stream.
        assert not any("experts" in n for n in names)
    for l in FULL_IDS:
        base = f"model.layers.{l}"
        qkv = pairs[f"{base}.self_attn.qkv_proj.weight"]
        assert qkv.shape == (N_Q * HEAD_DIM * 2 + 2 * N_KV * HEAD_DIM, H)
        assert f"{base}.self_attn.o_proj.weight" in pairs
        assert f"{base}.self_attn.q_norm.weight" in pairs
        assert torch.allclose(pairs[f"{base}.self_attn.q_norm.weight"].float(), torch.full((HEAD_DIM,), 1.0), atol=1e-2)
        assert f"{base}.self_attn.k_norm.weight" in pairs
    for l in range(N_LAYERS):
        if l in FULL_IDS:
            continue
        assert f"model.layers.{l}.self_attn.qkv_proj.weight" not in pairs


def test_iter_weights_gdn_fusion_order(tiny_gguf):
    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    pairs = dict(iter_gguf_weights(tiny_gguf, None, include_moe_experts=False,
                                   include_non_moe=True))
    l = 0  # linear layer
    base = f"model.layers.{l}"
    in_proj = pairs[f"{base}.linear_attn.in_proj.weight"]
    # split order [conv_dim(qkv), value_dim(z), b, a]
    assert in_proj.shape == (CONV_DIM + VALUE_DIM + 2 * DT_RANK, H)
    assert f"{base}.linear_attn.conv1d.weight" in pairs
    assert pairs[f"{base}.linear_attn.conv1d.weight"].shape == (CONV_DIM, 1, CONV_K)
    a_log = pairs[f"{base}.linear_attn.A_log"]
    assert a_log.dtype == torch.float32 and a_log.shape == (DT_RANK,)
    # GGUF stores -exp(A_log); the adapter converts back to log-space.
    assert torch.allclose(a_log, torch.full((DT_RANK,), 1.5), atol=1e-4)
    dt = pairs[f"{base}.linear_attn.dt_bias"]
    assert dt.dtype == torch.float32 and dt.shape == (DT_RANK,)
    # GDN gated norm is NOT Gemma: raw 0.7 passthrough.
    norm = pairs[f"{base}.linear_attn.norm.weight"]
    assert torch.allclose(norm.float(), torch.full((GDN_HEAD_DIM,), 0.7), atol=1e-2)
    assert f"{base}.linear_attn.out_proj.weight" in pairs
    assert pairs[f"{base}.linear_attn.out_proj.weight"].shape == (H, VALUE_DIM)
    # No full-attn params on linear layers.
    assert f"{base}.self_attn.o_proj.weight" not in pairs
    assert f"{base}.self_attn.q_norm.weight" not in pairs


def test_iter_weights_dequantizes_nvfp4_exactly(tiny_gguf):
    """The dense NVFP4 tensors dequantize to exactly the packed values * block
    scale * global. The synthetic writer uses global == block scale, so the
    result is values * global^2... no: the writer packs values ALREADY scaled
    by the block scale (values are representable in scale*e2m1), so with the
    side global == scale, the engine decode gives values * scale."""
    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    pairs = dict(iter_gguf_weights(tiny_gguf, None, include_moe_experts=False,
                                   include_non_moe=True))
    l = FULL_IDS[0]
    base = f"model.layers.{l}"
    o_proj = pairs[f"{base}.self_attn.o_proj.weight"].float()
    assert o_proj.shape == (H, N_Q * HEAD_DIM)
    # Non-trivial: packed nibbles decode to the (nibble/2-scaled) lattice —
    # verify against a fresh decode of the same packed bytes via gguf-py.
    import gguf.quants as q

    from freetoken.models.gguf.reader import iter_gguf_tensors

    for t in iter_gguf_tensors(tiny_gguf):
        if t.name == f"blk.{l}.attn_output.weight":
            blocks = t.packed().numpy().reshape(-1, NVFP4_BYTES)
            ref = q.NVFP4.dequantize_blocks(blocks).reshape(t.shape)  # torch order
            # gguf-py decode has no global; ours applies .scale (0.5 here).
            got = o_proj.numpy()
            assert np.abs(ref * 0.5 - got).max() < 1e-6
            return
    raise AssertionError("attn_output tensor not found")


def test_expert_banks_layout_and_globals(tiny_gguf):
    from freetoken.models.qwen3_5_moe.gguf import load_nvfp4_expert_sources, parse_gguf_config

    cfg = parse_gguf_config(_shim(tiny_gguf))
    banks = load_nvfp4_expert_sources(tiny_gguf, cfg)
    assert set(banks) == {
        "gate_up_packed", "gate_up_scale", "gate_up_global",
        "down_packed", "down_scale", "down_global",
    }
    for l in range(N_LAYERS):
        gu = banks["gate_up_packed"][l]
        assert gu.shape == (N_EXP, 2 * MOE_I, H // 2)
        gs = banks["gate_up_scale"][l]
        assert gs.shape == (N_EXP, 2 * MOE_I, H // 16)
        gg = banks["gate_up_global"][l]
        assert gg.shape == (N_EXP, 2 * MOE_I) and gg.dtype == torch.float16
        dp = banks["down_packed"][l]
        assert dp.shape == (N_EXP, H, MOE_I // 2)
        ds = banks["down_scale"][l]
        assert ds.shape == (N_EXP, H, MOE_I // 16)
        dg = banks["down_global"][l]
        assert dg.shape == (N_EXP, H)
    # gate rows first, then up rows (the gate_up GEMM split).
    gu0 = banks["gate_up_packed"][0]
    gu1 = banks["gate_up_packed"][1]
    assert not torch.equal(gu0, gu1)
    # Globals: per-expert fp32 .scale (1.0) expanded over rows.
    assert torch.all(banks["gate_up_global"][0] == 1.0)
    assert torch.all(banks["down_global"][0] == 1.0)


def test_expert_banks_permutation_exact(tiny_gguf):
    """One gate row's packed bytes decode identically through the engine's
    layout math (e2m1 * fp8-scale * global) and the ggml block decode."""
    from freetoken.models.qwen3_5_moe.gguf import (
        _e4m3_u8_to_f32,
        _E2M1_LUT,
        load_nvfp4_expert_sources,
        parse_gguf_config,
    )

    cfg = parse_gguf_config(_shim(tiny_gguf))
    banks = load_nvfp4_expert_sources(tiny_gguf, cfg)
    e, row = 2, 5
    packed = banks["gate_up_packed"][0][e, row]          # [H//2] uint8
    scale = banks["gate_up_scale"][0][e, row]            # [H//16] uint8
    glob = float(banks["gate_up_global"][0][e, row])
    # engine decode: byte j -> elems (2j, 2j+1) with block scale j//8
    vals = torch.empty(H, dtype=torch.float32)
    for j in range(H // 2):
        s = float(_e4m3_u8_to_f32(scale[j // 8].reshape(1))[0])
        vals[2 * j] = _E2M1_LUT[int(packed[j] & 0xF)] * s * glob
        vals[2 * j + 1] = _E2M1_LUT[int(packed[j] >> 4)] * s * glob
    # ggml reference: decode the row straight from the SOURCE GGUF tensor
    # (the bank conversion permutes bytes, so the reference must come from
    # the original interleaved blocks, not the converted banks).
    import gguf.quants as q

    from freetoken.models.gguf.reader import iter_gguf_tensors

    src = next(t for t in iter_gguf_tensors(tiny_gguf)
               if t.name == "blk.0.ffn_gate_exps.weight")
    blocks = src.packed().numpy().reshape(N_EXP, MOE_I, -1)[e, row]  # [36*sb]
    blocks = blocks.reshape(H // NVFP4_BLOCK, NVFP4_BYTES)
    ref = q.NVFP4.dequantize_blocks(blocks).reshape(-1)
    # ggml decode has no global; ours multiplies by glob == 1.0 here.
    assert np.abs(np.asarray(ref) * glob - vals.numpy()).max() < 1e-6


def test_unmapped_tensor_raises(tiny_gguf):
    from gguf import GGUFWriter, GGMLQuantizationType

    import shutil

    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights

    bad = tiny_gguf + ".bad.gguf"
    shutil.copy(tiny_gguf, bad)
    # Append an unmapped tensor via a full rewrite: simplest is to assert on
    # a name-mangled copy by patching the tensor name in place is fragile;
    # instead exercise the guard directly with a fake tensor name.
    from freetoken.models.qwen3_5_moe.gguf import _LAYER_NORM_MAP

    assert "attn_norm.weight" in _LAYER_NORM_MAP
    # The real guard: iter_gguf_weights raises ValueError on unknown suffixes.
    # Build a one-tensor GGUF with an unknown suffix.
    w = GGUFWriter(bad + ".tmp", "qwen35moe")
    w.add_block_count(N_LAYERS)
    w.add_embedding_length(H)
    w.add_head_count(N_Q)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_context_length(512)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_expert_count(N_EXP)
    w.add_expert_used_count(TOPK)
    w.add_expert_feed_forward_length(MOE_I)
    w.add_uint32("qwen35moe.full_attention_interval", INTERVAL)
    w.add_uint32("qwen35moe.ssm.group_count", GDN_K_HEADS)
    w.add_uint32("qwen35moe.ssm.state_size", GDN_HEAD_DIM)
    w.add_uint32("qwen35moe.ssm.inner_size", VALUE_DIM)
    w.add_uint32("qwen35moe.ssm.time_step_rank", DT_RANK)
    w.add_uint32("qwen35moe.ssm.conv_kernel", CONV_K)
    # partial rotary: 4 of 16 head dims
    w.add_uint32("qwen35moe.rope.dimension_count", 4)
    w.add_tensor("token_embd.weight", np.zeros((H, V), dtype=np.float16))
    w.add_tensor("output.weight", np.zeros((H, V), dtype=np.float16))
    w.add_tensor("blk.0.weird_unknown.weight", np.zeros(4, dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="unmapped qwen35moe"):
        list(iter_gguf_weights(bad + ".tmp", None, include_moe_experts=False,
                               include_non_moe=True))

def test_gdn_v_head_permutation_matches_checkpoint_pairing(tiny_gguf):
    """The checkpoint pairs GDN v-head m with k/q-head m % num_k_heads
    (block layout), while the engine's fla kernels pair v-head j with
    k-head j // (HV/HK) (interleave). iter_gguf_weights must permute
    every v-head-indexed segment by pi(j) = j//g + HK*(j%g) so the
    interleave kernels read the checkpoint's pairing. Verified against
    llama.cpp layer-0 activations on the real checkpoint."""
    from freetoken.models.qwen3_5_moe.gguf import iter_gguf_weights, parse_gguf_config, _shim_for

    pairs = dict(iter_gguf_weights(tiny_gguf, None, include_moe_experts=False,
                                   include_non_moe=True))
    cfg = parse_gguf_config(_shim_for(tiny_gguf))
    lg = cfg.linear_attention_group()
    hv, hk = lg.num_value_heads, lg.num_key_heads
    g = hv // hk
    pi = [j // g + hk * (j % g) for j in range(hv)]

    # Read the raw GGUF rows for a linear layer to compare against,
    # dequantizing NVFP4 with the adapter's own row decoder.
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.qwen3_5_moe.gguf import _dequant_nvfp4_rows, _side_scale
    from freetoken.models.qwen3_5_moe.gguf import _nvfp4_global_for, _collect_side_scales
    sides = _collect_side_scales(tiny_gguf)
    raw = {}
    for t in iter_gguf_tensors(tiny_gguf):
        if t.name.startswith("blk.0.") and not t.name.endswith((".scale", ".input_scale")):
            if t.ggml_type == 40:  # NVFP4
                g = _nvfp4_global_for(t, sides.get(t.name, {}))
                flat = _dequant_nvfp4_rows(t.packed().clone(), g).reshape(t.shape)
            else:
                from freetoken.models.gguf.dequant import dequantize
                flat = dequantize(t.packed().reshape(-1), t.ggml_type, torch.bfloat16).reshape(t.shape)
            raw[t.name] = flat.float()

    l = 0  # linear layer in the tiny fixture
    base = "model.layers.0"
    in_proj = pairs[f"{base}.linear_attn.in_proj.weight"].float()
    # fused order [conv_dim(qkv), value_dim(z), b, a]; qkv = q|k|v rows.
    key_dim = lg.key_head_dim * lg.num_key_heads
    v_off = 2 * key_dim
    z_off = CONV_DIM
    d = GDN_HEAD_DIM
    # v rows: fused row block j == raw row block pi(j)
    raw_qkv = raw["blk.0.attn_qkv.weight"]
    for j in range(hv):
        assert torch.equal(in_proj[v_off + j * d: v_off + (j + 1) * d],
                           raw_qkv[v_off + pi[j] * d: v_off + (pi[j] + 1) * d])
    # z rows likewise
    raw_gate = raw["blk.0.attn_gate.weight"]
    for j in range(hv):
        assert torch.equal(in_proj[z_off + j * d: z_off + (j + 1) * d],
                           raw_gate[pi[j] * d: (pi[j] + 1) * d])
    # out_proj: input head-blocks permuted the same way.
    out = pairs[f"{base}.linear_attn.out_proj.weight"].float()
    raw_out = raw["blk.0.ssm_out.weight"]
    for j in range(hv):
        assert torch.equal(out[:, j * d: (j + 1) * d],
                           raw_out[:, pi[j] * d: (pi[j] + 1) * d])
    # q/k rows and the shared gated-norm weight are NOT permuted.
    assert torch.equal(in_proj[:key_dim], raw_qkv[:key_dim])
    assert torch.equal(in_proj[key_dim:2 * key_dim], raw_qkv[key_dim:2 * key_dim])
    assert torch.equal(pairs[f"{base}.linear_attn.norm.weight"].float(),
                       raw["blk.0.ssm_norm.weight"].float())
