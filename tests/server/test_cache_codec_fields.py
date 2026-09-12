"""KV codec fields in the cache surfaces (TCQ plan item 3.3).

Pins:
- compute_cache_pools reports kv_codec / kv_codec_tune / innerq_calibrated
  read off the live pool + config (turbo and f16 pools);
- /v1/cache/status geometry carries the same fields (cache_geometry with a
  fake frontend state — hermetic, no server start);
- the cache report renders the codec suffix only for non-f16 codecs, and f16
  output is unchanged (no codec mention at all).
"""

from __future__ import annotations

from types import SimpleNamespace


def _fake_engine(codec_pool=None, kv_codec_tune="none"):
    """Minimal engine for compute_cache_pools: a plain KV pool."""
    pool = codec_pool if codec_pool is not None else SimpleNamespace()
    return SimpleNamespace(
        num_pages=64,
        config=SimpleNamespace(
            page_size=16,
            cache_type="radix",
            kv_codec_tune=kv_codec_tune,
            model_config=SimpleNamespace(dsv4_args=None, has_swa_attention=False),
        ),
        kv_cache=pool,
        moe_offload_cache=None,
        linear_state_pool=None,
    )


def test_pools_report_f16_defaults():
    from freetoken.kvcache.cache_status import compute_cache_pools

    pools = compute_cache_pools(_fake_engine())
    assert pools["kv_codec"] == "f16"
    assert pools["kv_codec_tune"] == "none"
    assert pools["innerq_calibrated"] is False


def test_pools_report_turbo_codec_and_calibration():
    from freetoken.kvcache.cache_status import compute_cache_pools

    turbo = SimpleNamespace(codec="turbo4", is_turbo=True, innerq_calibrated=True)
    pools = compute_cache_pools(_fake_engine(turbo, kv_codec_tune="innerq"))
    assert pools["kv_codec"] == "turbo4"
    assert pools["kv_codec_tune"] == "innerq"
    assert pools["innerq_calibrated"] is True

    # armed (not yet finalized) turbo pool: codec reported, calibration False
    turbo_armed = SimpleNamespace(
        codec="turbo3_tcq", is_turbo=True, innerq_calibrated=False
    )
    pools = compute_cache_pools(_fake_engine(turbo_armed, kv_codec_tune="innerq"))
    assert pools["kv_codec"] == "turbo3_tcq"
    assert pools["innerq_calibrated"] is False


def _state(kv_codec=None, kv_codec_tune=None, pools=None):
    """Fake frontend state for cache_geometry."""
    cfg_kwargs = {"page_size": 16, "memory_ratio": 0.9}
    if kv_codec is not None:
        cfg_kwargs["kv_codec"] = kv_codec
    if kv_codec_tune is not None:
        cfg_kwargs["kv_codec_tune"] = kv_codec_tune
    return SimpleNamespace(
        maintenance_state="serving",
        last_rebuild=None,
        stats=SimpleNamespace(kv_total_pages=64, mamba_total_slots=0),
        cache_pools=pools or {},
        config=SimpleNamespace(
            model_config=SimpleNamespace(num_experts=0, num_moe_layers=0),
            **cfg_kwargs,
        ),
        unit_bytes={"kv_bytes_per_token": 2048},
    )


def test_cache_geometry_carries_codec_fields():
    from freetoken.server.api_server import cache_geometry

    geo = cache_geometry(
        _state("turbo4", "innerq", {"innerq_calibrated": True})
    )
    assert geo["kv_codec"] == "turbo4"
    assert geo["kv_codec_tune"] == "innerq"
    assert geo["innerq_calibrated"] is True

    # f16 server (no codec fields on config at all): defaults, no raise
    geo = cache_geometry(_state())
    assert geo["kv_codec"] == "f16"
    assert geo["kv_codec_tune"] == "none"
    assert geo["innerq_calibrated"] is False


def test_cache_report_codec_suffix_only_for_non_f16():
    from freetoken.cache_report import format_cache_status

    def _doc(codec=None, tune=None, calibrated=False):
        geometry = {
            "num_pages": 64,
            "page_size": 1,
            "moe_cache_size": 0,
            "num_mamba_slots": 0,
            "num_swa_pages": 0,
            "swa_page_size": 0,
            "num_experts": 0,
            "num_moe_layers": 0,
            "cache_budget_bytes": 0,
            "unit_bytes": {"kv_per_token": 132},
        }
        if codec:
            geometry["kv_codec"] = codec
        if tune:
            geometry["kv_codec_tune"] = tune
            geometry["innerq_calibrated"] = calibrated
        return {"state": "serving", "geometry": geometry}

    f16 = format_cache_status(_doc())
    assert "codec" not in f16  # f16 output unchanged: no codec mention

    turbo = format_cache_status(_doc("turbo4"))
    assert "codec=turbo4" in turbo

    tuned = format_cache_status(_doc("turbo4", "innerq", True))
    assert "codec=turbo4" in tuned
    assert "tune=innerq" in tuned and "calibrated" in tuned

    tuned_pending = format_cache_status(_doc("turbo3_tcq", "innerq", False))
    assert "codec=turbo3_tcq" in tuned_pending
    assert "calibrated" not in tuned_pending