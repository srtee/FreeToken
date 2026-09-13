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
    assert len(got) == 19
    # spot-check the shapes read off the checkpoint
    assert got["fc"].shape == (2048, 4096)
    assert got["layer.mlp.experts_gate_up_proj"].shape == (256, 1024, 2048)
    assert got["layer.mlp.experts_down_proj"].shape == (256, 2048, 512)
    assert got["layer.self_attn.q_proj"].shape == (8192, 2048)
    # every norm got the Gemma (1+w) bake: compare against the raw
    # checkpoint value shifted by exactly +1
    from safetensors import safe_open
    with safe_open(_CKPT + "model-00003-of-00003.safetensors", framework="pt") as f:
        for mtp_key, baked_key in [
            ("mtp.pre_fc_norm_embedding.weight", "pre_fc_norm_embedding"),
            ("mtp.layers.0.input_layernorm.weight", "layer.input_layernorm"),
            ("mtp.layers.0.self_attn.k_norm.weight", "layer.self_attn.k_norm"),
        ]:
            raw = f.get_tensor(mtp_key).float()
            assert torch.allclose(got[baked_key].float(), raw + 1.0, atol=1e-2), baked_key


def test_mtp_head_state_dict_matches_emitted_keys():
    from freetoken.models.qwen3_5_moe.mtp import MTPHead
    cfg = _config()
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    mtp = MTPHead(cfg, emb, head)
    sd = dict(mtp.state_dict().items())
    got = _iter_mtp()
    assert set(sd) == set(got), (
        f"module-only={sorted(set(sd) - set(got))} "
        f"emitted-only={sorted(set(got) - set(sd))}")
    for k, t in got.items():
        assert tuple(sd[k].shape) == tuple(t.shape), k


def test_mtp_head_forward_one_step():
    from freetoken.models.qwen3_5_moe.mtp import MTPHead
    cfg = _config()
    torch.manual_seed(0)
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16)
    mtp = MTPHead(cfg, emb, head)
    # production loads cast each tensor to the model param's dtype
    mtp.load_state_dict({k: v.to(torch.bfloat16) for k, v in _iter_mtp().items()})
    with torch.no_grad():
        h0 = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16)
        tok = torch.tensor([1234])
        pos = torch.tensor([10])
        carry = mtp.forward(h0, tok, pos)
        assert carry.shape == (1, cfg.hidden_size)
        assert torch.isfinite(carry).all()
        # logits through the (stub) head: finite, right shape
        logits = head(carry)
        assert logits.shape == (1, cfg.vocab_size)
        # determinism: same inputs -> identical carry
        carry2 = mtp.forward(h0, tok, pos)
        assert torch.equal(carry, carry2)