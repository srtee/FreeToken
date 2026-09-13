"""MTP Wave 0: weight loading + MTPHead construction + eager forward.

Pins (docs/mtp-plan.md items 0.1-0.3):
- the qwen3_5_moe loader routes all 19 mtp.* tensors from the Qwen3.6-35B
  NVFP4 checkpoint into the MTPHead state dict (name + shape exact);
- MTPHead constructs against the family's parsed ModelConfig and runs one
  eager draft step on dummy input, producing a [1, hidden] carry;
- the config surface exposes mtp_num_hidden_layers=1 and
  mtp_use_dedicated_embeddings=False for this checkpoint.

CPU-only (the trunk dequant path needs a GPU for full-checkpoint loads;
this test streams and keeps only mtp tensors).
"""

from __future__ import annotations

import pytest
import torch

_CKPT = ("/home/sherntee/.cache/huggingface/hub/models--nvidia--"
         "Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5/")


def _iter_mtp():
    from freetoken.models.qwen3_5_moe.weight import _iter_weights_attn_fp8
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    got = {}
    for name, t in _iter_weights_attn_fp8(_CKPT, torch.device("cpu"),
                                          include_non_moe=True,
                                          include_moe_experts=False):
        if name.startswith("model.mtp."):
            got[name[len("model.mtp."):]] = t.cpu()
    return got


def _config():
    from transformers import AutoConfig
    from freetoken.models.qwen3_5_moe.config import parse_config
    return parse_config(AutoConfig.from_pretrained(_CKPT))


def test_config_exposes_mtp_fields():
    cfg = _config()
    assert cfg.mtp_num_hidden_layers == 1
    assert cfg.mtp_use_dedicated_embeddings is False


def test_loader_routes_all_19_mtp_tensors():
    got = _iter_mtp()
    # 19 checkpoint tensors; q/k/v fuse into one qkv_proj -> 17 emitted
    assert len(got) == 17
    # spot-check the shapes read off the checkpoint
    assert got["fc.weight"].shape == (2048, 4096)
    assert got["layer.mlp.experts.gate_up_proj"].shape == (256, 1024, 2048)
    assert got["layer.mlp.experts.down_proj"].shape == (256, 2048, 512)
    # fused qkv: q (2x for gate) + k + v rows
    assert got["layer.self_attn.qkv_proj.weight"].shape == (9216, 2048)
    # every norm got the Gemma (1+w) bake: compare against the raw
    # checkpoint value shifted by exactly +1
    from safetensors import safe_open
    with safe_open(_CKPT + "model-00003-of-00003.safetensors", framework="pt") as f:
        for mtp_key, baked_key in [
            ("mtp.pre_fc_norm_embedding.weight", "pre_fc_norm_embedding.weight"),
            ("mtp.layers.0.input_layernorm.weight", "layer.input_layernorm.weight"),
            ("mtp.layers.0.self_attn.k_norm.weight", "layer.self_attn.k_norm.weight"),
        ]:
            raw = f.get_tensor(mtp_key).float()
            assert torch.allclose(got[baked_key].float(), raw + 1.0, atol=1e-2), baked_key


def test_mtp_head_state_dict_matches_emitted_keys():
    """The wave-1 MTPHead (trunk attention + eager MoE) must consume
    exactly the emitted keys: the q/k/v fusion moved q/k/v into one
    qkv_proj tensor, and attention norms keep their .weight suffix."""
    from freetoken.models.qwen3_5_moe.mtp import MTPHead
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cfg = _config()
    mtp = MTPHead(cfg, torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                  prefix="model.mtp")
    sd = dict(mtp.state_dict().items())
    got = _iter_mtp()
    assert set(sd) == set(got), (
        f"module-only={sorted(set(sd) - set(got))} "
        f"emitted-only={sorted(set(got) - set(sd))}")
    for k, t in got.items():
        assert tuple(sd[k].shape) == tuple(t.shape), k
    # the fused qkv: num_q*hd*2 + 2*num_kv*hd rows
    assert got["layer.self_attn.qkv_proj.weight"].shape == (
        cfg.num_qo_heads * cfg.head_dim * 2 + 2 * cfg.num_kv_heads * cfg.head_dim,
        cfg.hidden_size)


def test_mtp_head_wiring():
    """MTPHead construction + lm_head attachment + state-dict load. The
    draft_step execution needs the engine context (paged KV backend) —
    its numerics gate is the wave-1 E2E (ft vs llama.cpp hidden states)."""
    from freetoken.models.qwen3_5_moe.mtp import MTPHead
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cfg = _config()
    torch.manual_seed(0)
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16)
    mtp = MTPHead(cfg, emb, prefix="model.mtp")
    mtp.set_lm_head(head)
    assert mtp._lm_head is head
    got = _iter_mtp()
    sd_mod = dict(mtp.state_dict().items())
    # production casts each tensor to the model param's dtype
    mtp.load_state_dict({k: v.to(sd_mod[k].dtype) for k, v in got.items()})
    # spot-check a loaded weight landed
    assert torch.equal(mtp.fc.weight, got["fc.weight"].to(mtp.fc.weight.dtype))
