"""Pool-factory generalization: hybrid linear x ANY paged family.

The old factory hard-required "linear + one GQA group" (Qwen3.5 GDN shape);
glm5_next is linear (KDA) x DSA. Checks the factory dispatch, the MLA/DSA
layer-id remap (34 KDA layers cost no latent slabs), the kpool pool selection
(gate slab), and the KV cost model's kpool double-count of the index slabs.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache import create_kvcache_pool, resolve_pool_class
from freetoken.kvcache.dsa_pool import DSAKVCache, KpoolDSAKVCache
from freetoken.kvcache.mha_pool import MHAKVCache
from freetoken.kvcache.turbo_pool import TurboKVCache
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)


@pytest.fixture(autouse=True)
def _single_rank_tp():
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _glm5_like_config(index_kpool=4, index_head_dim=128):
    n_layers = 12
    dsa_ids = tuple(range(3, n_layers, 4))  # 3, 7, 11
    kda_ids = tuple(i for i in range(n_layers) if i not in dsa_ids)
    rotary = RotaryConfig(head_dim=256, rotary_dim=0, max_position=4096, base=1e4, scaling=None)
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=kda_ids,
            num_key_heads=4, num_value_heads=4, key_head_dim=128, value_head_dim=128,
            conv_kernel_dim=4, output_gate="sigmoid", variant="kda",
        ),
        FullAttentionGroupConfig(
            name="full", layer_ids=dsa_ids, num_kv_heads=1, head_dim=512,
            rotary_config=rotary, mla=True,
            index_head_dim=index_head_dim, num_index_layers=len(dsa_ids),
            index_ratio=index_kpool,
        ),
    )
    return ModelConfig(
        num_layers=n_layers, num_qo_heads=4, num_kv_heads=1, head_dim=512,
        hidden_size=256, vocab_size=1000, intermediate_size=512,
        rms_norm_eps=1e-5, rotary_config=rotary, hidden_act="silu",
        tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, norm_topk_prob=True, model_type="glm5_next",
        architectures=["Glm5NextForCausalLM"], moe_enabled=True,
        attention_groups=groups,
    )


def test_factory_builds_kpool_pool_with_layer_remap():
    cfg = _glm5_like_config()
    assert resolve_pool_class(cfg) is KpoolDSAKVCache

    pool = create_kvcache_pool(
        model_config=cfg, num_pages=4, page_size=64,
        dtype=torch.bfloat16, device=torch.device("cpu"), num_req_slots=5,
    )
    assert isinstance(pool, KpoolDSAKVCache)
    # Latent slabs back ONLY the 3 DSA layers (34-of-45 economy at real scale).
    assert pool._kv_buffer.shape[1] == 3
    # Global layer-id addressing: DSA layers resolve, KDA layers have no slab.
    for lid in (3, 7, 11):
        assert pool.latent_rows(lid).shape == (256, 512)
    with pytest.raises(KeyError):
        pool.latent_rows(0)  # a KDA layer
    # Shadow index slab: tokens/ratio rows + one scratch row per request slot.
    assert pool.index_k_cache(0).shape == (256 // 4 + 5, 128)
    assert pool.cmp_scratch_base == 256 // 4
    # kpool tail rings exist at [num_req_slots, ratio, head_dim] per indexer layer.
    assert pool.tail_k(0).shape == pool.tail_gate(0).shape
    assert pool.tail_k(0).shape == (5, 4, 128)


def test_factory_kpool1_builds_plain_dsa_pool():
    cfg = _glm5_like_config(index_kpool=1)
    assert resolve_pool_class(cfg) is DSAKVCache
    pool = create_kvcache_pool(
        model_config=cfg, num_pages=16, page_size=1,
        dtype=torch.bfloat16, device=torch.device("cpu"),
    )
    assert type(pool) is DSAKVCache


def test_cost_model_kpool_shadow_slab_quarter_cost():
    """The shadow slab stores one row per index_ratio tokens: the kpool spec's
    index bytes are 1/ratio of the plain DSA slab (rings/scratch are per-request
    and not part of the per-token price)."""
    from types import SimpleNamespace

    from freetoken.kvcache.base import spec_kv_bytes_per_token

    tp = SimpleNamespace(size=1)
    econf = SimpleNamespace(tp_info=tp, dtype=torch.bfloat16)
    (spec,) = [
        s for s in _glm5_like_config().kv_cache_group_specs() if s.num_layers > 0
    ]
    (spec1,) = [
        s
        for s in _glm5_like_config(index_kpool=1).kv_cache_group_specs()
        if s.num_layers > 0
    ]
    index_full = spec.index_head_dim * spec.num_index_layers * 2
    assert (
        spec_kv_bytes_per_token(spec1, econf) - spec_kv_bytes_per_token(spec, econf)
        == index_full - index_full // 4
    )


def _qwen35_like_config(mtp_layers=1):
    """Qwen3.6-35B shape: GDN hybrid x plain GQA full group, head_dim 256,
    MTP draft rows appended to full_ids at layer index num_layers.."""
    n_layers = 12
    full_ids = tuple(range(0, n_layers, 4))  # 0, 4, 8
    linear_ids = tuple(i for i in range(n_layers) if i not in full_ids)
    rotary = RotaryConfig(head_dim=256, rotary_dim=0, max_position=4096, base=1e4, scaling=None)
    if mtp_layers:
        # The draft head's KV rows live at index num_layers+i (qwen3_5_moe
        # config.py appends them to the full group's layer_ids).
        full_ids = full_ids + tuple(n_layers + i for i in range(mtp_layers))
    groups = (
        LinearGatedDeltaGroupConfig(
            name="linear", layer_ids=linear_ids,
            num_key_heads=4, num_value_heads=4, key_head_dim=128, value_head_dim=128,
            conv_kernel_dim=4, output_gate="sigmoid", variant="kda",
        ),
        FullAttentionGroupConfig(
            name="full", layer_ids=full_ids, num_kv_heads=2, head_dim=256,
            rotary_config=rotary,
        ),
    )
    return ModelConfig(
        num_layers=n_layers, num_qo_heads=8, num_kv_heads=2, head_dim=256,
        hidden_size=256, vocab_size=1000, intermediate_size=512,
        rms_norm_eps=1e-5, rotary_config=rotary, hidden_act="silu",
        tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, norm_topk_prob=True, model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"], moe_enabled=True,
        attention_groups=groups, mtp_num_hidden_layers=mtp_layers,
    )


def test_factory_turbo_pool_accepts_mtp_layer_ids():
    """The turbo branch passed the bare trunk layer count as the pool's
    global-id bound, so a hybrid-linear model with MTP draft rows in
    full_ids raised 'KV layer id 12 outside [0, 12)' at construction while
    the f16 path (which widens the bound) served the same checkpoint."""
    cfg = _qwen35_like_config()

    pool = create_kvcache_pool(
        model_config=cfg, num_pages=4, page_size=1,
        dtype=torch.float16, device=torch.device("cpu"), kv_codec="turbo8",
    )
    assert isinstance(pool, TurboKVCache)
    # Packed slabs back exactly the paged ids (3 trunk + 1 MTP draft row),
    # never the linear layers and never num_layers + mtp.
    assert pool._k_packed.shape[0] == 4
    assert pool._v_packed.shape[0] == 4
    # The MTP draft row resolves through the dense remap (bound covers it).
    assert pool._dense(cfg.num_layers) == 3

    # The f16 path keeps the same widened bound for the same checkpoint.
    f16_pool = create_kvcache_pool(
        model_config=cfg, num_pages=4, page_size=1,
        dtype=torch.float16, device=torch.device("cpu"),
    )
    assert type(f16_pool) is MHAKVCache
    assert f16_pool.num_layers == cfg.num_layers + 1
    assert f16_pool._dense(cfg.num_layers) == 3
    # Storage still counts only paged ids.
    assert f16_pool._kv_buffer.shape[1] == 4


def test_factory_turbo_pool_without_mtp_unchanged():
    """No MTP rows -> the bound is the trunk count and turbo dispatch is
    unchanged (regression guard for the widened-bound hoist)."""
    cfg = _qwen35_like_config(mtp_layers=0)
    pool = create_kvcache_pool(
        model_config=cfg, num_pages=4, page_size=1,
        dtype=torch.float16, device=torch.device("cpu"), kv_codec="turbo8",
    )
    assert isinstance(pool, TurboKVCache)
    assert pool.num_layers == cfg.num_layers
    assert pool._k_packed.shape[0] == 3