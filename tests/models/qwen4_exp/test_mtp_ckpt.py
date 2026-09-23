"""Real-checkpoint mapping test for the Flash-Next MTP draft head.

Gated on ``FREETOKEN_QWEN4EXP_MODEL`` (the RadixArk/Qwen3.8-Flash-Next-NVFP4
download). Reads only the three MTP-bearing trunk shards through safetensors'
lazy reader, replays the production loader remap over them and strict-loads
the result into :class:`Qwen4ExpMTPHead`: any key, fusion or shape drift in
the port fails here instead of deep inside a serve's weight load.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
from freetoken.models.qwen4_exp.weight import iter_weights
from freetoken.utils import cached_load_hf_config

from .test_mtp import _EmbedStub

MODEL_PATH = os.environ.get("FREETOKEN_QWEN4EXP_MODEL")
pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.skipif(not MODEL_PATH, reason="FREETOKEN_QWEN4EXP_MODEL is not set"),
]

# The MTP tensors live at the tail of the trunk: one in -00010, 26 in -00011,
# 4 in -00012 (the extracted checkpoint inventory).
MTP_SHARDS = (
    "model-bf16-00010.safetensors",
    "model-bf16-00011.safetensors",
    "model-bf16-00012.safetensors",
)


@pytest.fixture(scope="module")
def config():
    return parse_config(cached_load_hf_config(MODEL_PATH))


@pytest.fixture(scope="module")
def mtp_state(tmp_path_factory, config):
    """``model.mtp.*`` tensors exactly as ``iter_weights`` remaps them.

    A shadow dir with symlinks to just the MTP-bearing shards (plus the real
    config.json, which gates the remap) keeps the loader walk off the other
    ~110 GB of the checkpoint.
    """
    shadow = tmp_path_factory.mktemp("mtp-shards")
    shutil.copy(os.path.join(MODEL_PATH, "config.json"), shadow / "config.json")
    for shard in MTP_SHARDS:
        os.symlink(os.path.join(MODEL_PATH, shard), shadow / shard)
    state = {
        name: tensor
        for name, tensor in iter_weights(
            str(shadow), torch.device("cpu"),
            include_moe_experts=False, include_non_moe=True,
        )
        if name.startswith("model.mtp.")
    }
    assert len(state) == 31, sorted(state)
    return state


def test_mtp_key_coverage_matches_the_module(mtp_state, config):
    head = Qwen4ExpMTPHead(config, _EmbedStub(config))
    assert set(mtp_state) == set(head.state_dict(prefix="model.mtp"))


def test_mtp_state_strict_loads_and_is_healthy(mtp_state, config):
    head = Qwen4ExpMTPHead(config, _EmbedStub(config))
    head.load_state_dict(mtp_state, prefix="model.mtp")
    for key, tensor in mtp_state.items():
        assert tensor.dtype == torch.bfloat16, key  # the head ships plain bf16
        assert torch.isfinite(tensor.float()).all(), key


def test_mtp_neck_finite_on_real_weights(mtp_state, config):
    head = Qwen4ExpMTPHead(config, _EmbedStub(config))
    head.load_state_dict(mtp_state, prefix="model.mtp")
    T = 2
    carry = torch.randn(T, config.mtp_hidden_size, dtype=torch.bfloat16)
    fused = head._fuse_neck(torch.randn(T, config.hidden_size, dtype=torch.bfloat16), carry)
    assert fused.shape == (T, config.mtp_hidden_size)
    assert torch.isfinite(fused.float()).all()
