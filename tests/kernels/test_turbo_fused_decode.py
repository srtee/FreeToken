"""Wave-2 fused decode: Triton split-k decode straight off the packed
slabs, with the dequant inverse-FWHT folded into the query prologue and
the stage-2 epilogue (the per-KV-row work is pure unpack + dot).

Pins, per codec:
- fused decode output == materializer decode path (decode_paged_attention
  over pool.materialize rows) within fp16 tolerance;
- the fold identities hold at head_dim 128 and 256;
- stage-2 combines rotated-domain partials correctly (max_kv_splits > 1).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.attention import (
    decode_paged_attention,
    decode_paged_attention_turbo,
)


CODECS = ["turbo8", "turbo4", "turbo3_tcq"]


@pytest.fixture(name="pool_and_data")
def pool_fixture(request):
    codec, head_dim = request.param
    from freetoken.kvcache.turbo_pool import TurboKVCache
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    torch.manual_seed(20)
    hv, n_pages, batch = 2, 256, 3
    pool = TurboKVCache(
        num_kv_heads=hv, num_layers=1, head_dim=head_dim, num_pages=n_pages,
        page_size=1, dtype=torch.float16, device=torch.device("cuda"),
        codec=codec, layer_ids=[0],
    )
    seq = [37, 64, 129]
    loc = torch.arange(sum(seq), dtype=torch.int32, device="cuda")
    k = (torch.randn(sum(seq), hv, head_dim, device="cuda") * 0.5)
    v = (torch.randn(sum(seq), hv, head_dim, device="cuda") * 0.5)
    pool.store_kv(k, v, loc, layer_id=0)

    num_q_heads = hv * 4
    q = (torch.randn(batch, num_q_heads, head_dim, device="cuda") * 0.5).half()
    indptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq), 0)),
                          dtype=torch.int32, device="cuda")
    indices = torch.arange(sum(seq), dtype=torch.int32, device="cuda")
    q_pos = torch.tensor([s - 1 for s in seq], dtype=torch.int64, device="cuda")
    max_splits = 4
    logits = torch.zeros(batch, num_q_heads, max_splits, head_dim,
                         dtype=torch.float32, device="cuda")
    lse = torch.zeros(batch, num_q_heads, max_splits,
                      dtype=torch.float32, device="cuda")
    splits = torch.full((batch,), max_splits, dtype=torch.int32, device="cuda")
    aux = pool.decode_aux()
    return pool, codec, head_dim, q, indptr, indices, q_pos, logits, lse, splits, max_splits, aux, seq


def _materializer_reference(pool, codec, head_dim, q, indptr, indices, q_pos,
                            logits, lse, splits, max_splits, aux, seq):
    """The eager wave-1 path: materialize rows then generic triton decode."""
    pt = torch.arange(sum(seq), dtype=torch.int32, device="cuda")
    k_m, v_m = pool.materialize(layer_id=0, page_table=pt, cache_seqlens=None)
    return decode_paged_attention(
        q=q, k_cache=k_m, v_cache=v_m, indptr=indptr, indices=indices,
        q_positions=q_pos, attn_logits=logits, attn_lse=lse,
        num_kv_splits=splits, max_kv_splits=max_splits,
        sm_scale=head_dim ** -0.5,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("pool_and_data", [
    (c, hd) for c in CODECS for hd in (128, 256)
], indirect=True)
def test_fused_decode_matches_materializer(pool_and_data):
    pool, codec, head_dim, q, indptr, indices, q_pos, logits, lse, splits, max_splits, aux, seq = pool_and_data
    ref = _materializer_reference(pool, codec, head_dim, q, indptr, indices,
                                  q_pos, logits, lse, splits, max_splits, aux, seq)
    # fresh scratch for the fused call (decode kernels write into them)
    logits2 = torch.zeros_like(logits)
    lse2 = torch.zeros_like(lse)
    dense = pool._dense(0)
    got = decode_paged_attention_turbo(
        q=q,
        k_slab=pool._k_packed[dense],
        v_slab=pool._v_packed[dense],
        indptr=indptr, indices=indices, q_positions=q_pos,
        attn_logits=logits2, attn_lse=lse2, num_kv_splits=splits,
        max_kv_splits=max_splits, sm_scale=head_dim ** -0.5,
        aux=aux, codec=codec,
    )
    # the fused kernel computes in fp32 throughout (ieee dot), the
    # reference materializes to fp16 then decodes in fp16 kernels —
    # tolerance covers the fp16 rounding of the two paths
    assert got.shape == ref.shape
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(0.25)
    rel = (diff / denom)
    assert rel.mean() < 2e-2, f"{codec} hd{head_dim}: mean rel {rel.mean():.4f}"
    assert rel.max() < 1e-1, f"{codec} hd{head_dim}: max rel {rel.max():.4f}"
