"""MTP head loading against a synthetic RadixArk-shaped checkpoint.

The inverse of the loader contract: the module tree defines the state dict, the
test derives the checkpoint's raw ``mtp.*`` keys from it (unfusing the fused
projections), writes them as a tiny safetensors dir, and requires
``iter_weights`` to reproduce the tree exactly under a strict load. Also pins
the config gate: without a ``config.json`` declaring MTP layers, the loader
must keep dropping ``mtp.*`` (the pre-port behavior), and the fc_hidden /
fc_embedding neck must compute the sglang reference formula.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
from freetoken.models.qwen4_exp.weight import iter_weights

from .common import parsed_config


@pytest.fixture()
def mtp_config():
    return parsed_config(mtp_num_hidden_layers=1)


class _EmbedStub:
    def __init__(self, config) -> None:
        self._hidden = config.hidden_size

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.numel(), self._hidden)


def _raw_from_state(state: dict[str, torch.Tensor], cfg) -> dict[str, torch.Tensor]:
    """Unfuse the module tree into checkpoint-shaped raw ``mtp.*`` keys."""
    q_rows = 2 * cfg.num_qo_heads * cfg.head_dim
    kv_rows = cfg.num_kv_heads * cfg.head_dim
    lr = cfg.qwen4_args.hc_lowrank
    hc = cfg.qwen4_args.hc_count
    raw: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        assert key.startswith("model.mtp."), key
        sub = key[len("model.mtp."):]
        if sub.endswith("self_attn.qkv_proj.weight"):
            base = sub[: -len("qkv_proj.weight")]
            raw[f"mtp.{base}q_proj.weight"] = tensor[:q_rows]
            raw[f"mtp.{base}k_proj.weight"] = tensor[q_rows:q_rows + kv_rows]
            raw[f"mtp.{base}v_proj.weight"] = tensor[q_rows + kv_rows:]
        elif sub.endswith("input_mix_weight_down_block_inject.weight"):
            base = sub[: -len("input_mix_weight_down_block_inject.weight")]
            raw[f"mtp.{base}input_mix_weight_down.weight"] = tensor[:lr]
            raw[f"mtp.{base}block_inject_weight.weight"] = tensor[lr:lr + hc]
        elif sub.endswith("shared_expert.gate_up_proj.weight"):
            base = sub[: -len("gate_up_proj.weight")]
            half = tensor.shape[0] // 2
            raw[f"mtp.{base}gate_proj.weight"] = tensor[:half]
            raw[f"mtp.{base}up_proj.weight"] = tensor[half:]
        elif sub.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
            raw[f"mtp.{sub}.weight"] = tensor
        else:
            raw[f"mtp.{sub}"] = tensor
    return raw


def test_mtp_loader_roundtrip(tmp_path, mtp_config):
    head = Qwen4ExpMTPHead(mtp_config, _EmbedStub(mtp_config))
    state = head.state_dict(prefix="model.mtp")  # the production load path
    assert len(state) > 20  # the full head, not a stub

    raw = _raw_from_state(state, mtp_config)
    (tmp_path / "config.json").write_text(json.dumps({
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }))
    save_file(raw, str(tmp_path / "model.safetensors"))

    loaded = dict(iter_weights(str(tmp_path), torch.device("cpu"),
                               include_moe_experts=False, include_non_moe=True))
    assert set(loaded) == set(state)
    for key, tensor in state.items():
        assert loaded[key].shape == tensor.shape, key
        assert loaded[key].dtype == tensor.dtype, key
    head.load_state_dict(loaded, prefix="model.mtp")  # strict: any mismatch raises


def test_mtp_gate_drops_keys_without_config(tmp_path, mtp_config):
    head = Qwen4ExpMTPHead(mtp_config, _EmbedStub(mtp_config))
    raw = _raw_from_state(head.state_dict(prefix="model.mtp"), mtp_config)
    save_file(raw, str(tmp_path / "model.safetensors"))  # no config.json
    emitted = [k for k, _ in iter_weights(str(tmp_path), torch.device("cpu"),
                                          include_moe_experts=False, include_non_moe=True)
               if k.startswith("model.mtp.")]

def test_mtp_neck_math(mtp_config):
    head = Qwen4ExpMTPHead(mtp_config, _EmbedStub(mtp_config))
    # torch.empty buffers hold garbage: seed before any forward (a NaN/inf bit
    # pattern flips inf<->NaN across BLAS reduction orders and breaks compare).
    torch.manual_seed(0)
    for tensor in head.state_dict().values():
        tensor.normal_(0.0, 0.02)
    T = 3
    H = mtp_config.hidden_size
    hc = mtp_config.qwen4_args.hc_count
    emb = torch.randn(T, H, dtype=torch.float32)
    carry = torch.randn(T, hc * H, dtype=torch.float32)

    out = head._fuse_neck(emb, carry)

    # sglang _fuse_hc_input: the fc_embedding(norm_e) broadcast is ADDED to the
    # fc_hidden(norm_h) per-stream block (not concatenated), stream-major.
    expect = (
        head.fc_embedding.forward(head.pre_fc_norm_embedding.forward(emb)).unsqueeze(1)
        + head.fc_hidden.forward(head.pre_fc_norm_hidden.forward(carry).view(T, hc, H))
    ).reshape(T, hc * H)
    assert out.shape == (T, hc * H)
    torch.testing.assert_close(out, expect)
