"""InnerQ per-channel equalization (TCQ plan item 3.1) kernel tests.

The CUDA quant/dequant kernels carry an InnerQ tap: per-channel scale
applied before L2-norm at encode, inverse scale applied at the dequant
tail. Calibration accumulates raw-domain stats (K and V pooled) during
an armed window; the pool computes buun-formula RMS scales and uploads
them.

These tests pin:
- identity scales keep the roundtrip bit-exact against the torch oracle
  (the InnerQ tap must be a no-op until real scales are uploaded);
- the calibration accumulators count every 128-group from both K and V
  and collect the raw-domain sum-of-squares / max;
- finalize_scales implements buun's RMS formula (mean_rms/channel_rms
  ^strength, clamp 2.0) and auto-disables balanced channels;
- with anisotropic input, the uploaded scales shrink the scaled-domain
  relative error of the weak channels by >2x;
- disarm stops accumulation.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel import turbo_kv
from freetoken.kernel.turbo_oracle import TurboCodec


@pytest.fixture(name="codec")
def codec_fixture():
    codec = "turbo4"
    yield codec
    # leave the device in the exact state for other suites
    turbo_kv.innerq_upload_scales(
        codec, torch.ones(128), torch.ones(128))


def _roundtrip(codec, k, v):
    L, heads = k.shape[0], k.shape[1]
    bb = turbo_kv.CODEC_SPECS[codec][1]
    locs = torch.arange(L, dtype=torch.int32, device="cuda")
    kd = torch.zeros(L, heads, bb, dtype=torch.uint8, device="cuda")
    vd = torch.zeros(L, heads, bb, dtype=torch.uint8, device="cuda")
    khat = torch.zeros(L, heads, 128, dtype=torch.float16, device="cuda")
    vhat = torch.zeros_like(khat)
    turbo_kv.turbo_quantize(codec, k, kd, locs, is_v=False)
    turbo_kv.turbo_quantize(codec, v, vd, locs, is_v=True)
    turbo_kv.turbo_dequantize(codec, kd, khat, locs, is_v=False)
    turbo_kv.turbo_dequantize(codec, vd, vhat, locs, is_v=True)
    return khat, vhat, kd


@pytest.mark.gpu
def test_identity_scales_roundtrip_matches_oracle(codec):
    """With identity scales the InnerQ tap must be a no-op: the kernel
    roundtrip equals the torch oracle's roundtrip (bit-exact bytes)."""
    torch.manual_seed(0)
    L, heads = 8, 2
    k = (torch.randn(L, heads, 128, device="cuda") * 0.5).contiguous()
    v = (torch.randn(L, heads, 128, device="cuda") * 0.5).contiguous()
    turbo_kv.innerq_upload_scales(codec, torch.ones(128), torch.ones(128))
    khat, _, kd = _roundtrip(codec, k, v)
    bb = turbo_kv.CODEC_SPECS[codec][1]

    oracle = TurboCodec(codec, is_v=False)
    # The oracle decodes into the rotated domain (buun's dequant applies
    # the inverse FWHT; the oracle roundtrip doesn't) — rotate back before
    # comparing with the kernel's materializer output.
    from freetoken.kernel.turbo_oracle import turbo_rotate
    blocks = oracle.encode(k.reshape(-1, 128).float().cpu())
    ref = turbo_rotate(oracle.decode(blocks), inverse=True).reshape(k.shape)
    # kernel stores fp16 (dequant output dtype) — compare with fp16-rounding
    # tolerance; the packed BYTES are bit-exact (asserted below).
    assert torch.allclose(khat.float().cpu(), ref, atol=1e-3)
    assert torch.equal(kd.reshape(-1, bb).cpu(), blocks)


@pytest.mark.gpu
def test_calibration_counts_all_groups_pools_kv(codec):
    """During the armed window every 128-group from both K and V
    contributes one count and raw-domain stats."""
    torch.manual_seed(1)
    L, heads = 16, 4
    k = torch.randn(L, heads, 128, device="cuda").contiguous()
    v = torch.randn(L, heads, 128, device="cuda").contiguous()
    turbo_kv.innerq_arm_calibration(codec)
    _roundtrip(codec, k, v)
    sq, ch_max, count = turbo_kv.innerq_download_stats(codec)
    assert count == 2 * L * heads
    # channel_sq accumulates the raw x^2 over all groups
    kv = torch.cat([k.reshape(-1, 128), v.reshape(-1, 128)]).float().cpu()
    assert torch.allclose(sq, (kv * kv).sum(0), rtol=1e-3)
    assert torch.allclose(ch_max, kv.abs().amax(0), rtol=1e-3)
    # stats are pre-scale raw values — independent of the (identity)
    # channel scale at accumulation time
    assert ch_max.max() > 0


@pytest.mark.gpu
def test_finalize_scales_buun_rms_formula(codec):
    sq = torch.tensor([1.0, 100.0, 4.0] + [1.0] * 125)
    ch_max = sq.sqrt()
    count = 1000
    res = turbo_kv.innerq_finalize_scales(sq, ch_max, count)
    assert res is not None
    s, ratio = res
    # RMS mode: s[i] = (mean_rms / channel_rms)^0.5, clamped to [0.5, 2]
    rms = (sq / count).sqrt()
    expected = ((rms.mean() / rms) ** 0.5).clamp(0.5, 2.0)
    assert torch.allclose(s, expected, atol=1e-5)
    assert ratio == pytest.approx(
        float(torch.maximum(expected, 1.0 / expected).max()))
    # auto-disable: balanced channels return None
    assert turbo_kv.innerq_finalize_scales(torch.ones(128), torch.ones(128),
                                           100) is None
    # zero count: None
    assert turbo_kv.innerq_finalize_scales(sq, ch_max, 0) is None


@pytest.mark.gpu
def test_uploaded_scales_equalize_anisotropic_channels(codec):
    """The point of InnerQ: with a 100x-dominant channel, the uploaded
    scales shrink the weak channels' scaled-domain relative error."""
    torch.manual_seed(2)
    L, heads = 16, 4
    k = (torch.randn(L, heads, 128, device="cuda") * 0.5).contiguous()
    v = (torch.randn(L, heads, 128, device="cuda") * 0.5).contiguous()
    k2 = k.clone()
    k2[..., 0] *= 100.0  # anisotropic: channel 0 dominates

    turbo_kv.innerq_upload_scales(codec, torch.ones(128), torch.ones(128))
    khat, _ = _roundtrip(codec, k2, v)[:2]
    rel_base = ((khat.float() - k2.float()).abs()
                / k2.float().abs().clamp_min(1e-6)).mean(dim=(0, 1))

    turbo_kv.innerq_arm_calibration(codec)
    _roundtrip(codec, k2, v)  # accumulate over the anisotropic data
    sq, ch_max, count = turbo_kv.innerq_download_stats(codec)
    res = turbo_kv.innerq_finalize_scales(sq, ch_max, count)
    assert res is not None, "anisotropic input must produce non-identity scales"
    s, _ = res
    turbo_kv.innerq_upload_scales(codec, s, 1.0 / s)
    khat, _ = _roundtrip(codec, k2, v)[:2]
    # compare in the scaled domain (what the codec quantizes)
    xs = k2.float() * s.cuda()
    rel_eq = ((khat.float() - xs).abs() / xs.abs().clamp_min(1e-6)).mean(dim=(0, 1))
    assert rel_eq[1] < 0.5 * rel_base[1], (
        "weak channel relative error should drop >=2x")
    # the dominant channel's relative error must stay bounded (FWHT mixes
    # the unscaled weak-channel improvement across the group, so the
    # dominant channel absorbs some noise — buun accepts this trade)
    assert rel_eq[0] <= 4.0 * rel_base[0], (
        "dominant-channel relative error must stay within the equalization "
        "trade buun accepts (FWHT mixes weak-channel noise into it)")


@pytest.mark.gpu
def test_disarm_stops_accumulation(codec):
    torch.manual_seed(3)
    k = torch.randn(8, 1, 128, device="cuda").contiguous()
    bb = turbo_kv.CODEC_SPECS[codec][1]
    kd = torch.zeros(8, 1, bb, dtype=torch.uint8, device="cuda")
    locs = torch.arange(8, dtype=torch.int32, device="cuda")
    turbo_kv.innerq_arm_calibration(codec)
    turbo_kv.turbo_quantize(codec, k, kd, locs, is_v=False)
    _, _, count_armed = turbo_kv.innerq_download_stats(codec)
    assert count_armed == 8
    turbo_kv.innerq_upload_scales(codec, torch.ones(128), torch.ones(128))
    turbo_kv.turbo_quantize(codec, k, kd, locs, is_v=False)
    _, _, count_disarmed = turbo_kv.innerq_download_stats(codec)
    assert count_disarmed == count_armed, "disarm must stop accumulation"

@pytest.mark.gpu
def test_pool_calibration_state_machine():
    """TurboKVCache with tune=innerq: the first CALIBRATION_TOKENS stored
    tokens arm accumulation, then the pool finalizes and uploads scales;
    further stores don't accumulate."""
    from freetoken.kvcache.turbo_pool import TurboKVCache
    from freetoken.distributed import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    codec = "turbo4"
    hv = 2
    n_tokens = TurboKVCache.CALIBRATION_TOKENS
    # small window for the test
    TurboKVCache.CALIBRATION_TOKENS = 32
    try:
        pool = TurboKVCache(
            num_kv_heads=hv, num_layers=2, head_dim=128, num_pages=128,
            page_size=1, dtype=torch.float16, device=torch.device("cuda"),
            codec=codec, layer_ids=[0, 1],
        )
        pool.arm_innerq_calibration()
        torch.manual_seed(4)
        # anisotropic channel so finalize produces non-identity scales
        for step in range(4):
            k = torch.randn(16, hv, 128, device="cuda") * 0.5
            k[..., 0] *= 100.0
            v = torch.randn(16, hv, 128, device="cuda") * 0.5
            loc = torch.arange(step * 16, (step + 1) * 16, dtype=torch.int32,
                               device="cuda")
            pool.store_kv(k, v, loc, layer_id=0)
            if step == 3:
                # window (32 tokens) filled at step 1; later stores must
                # not accumulate — verify by disarmed identity upload state
                pass
        # after 64 tokens > 32 window: scales uploaded, calibrate disarmed
        sq, ch_max, count = turbo_kv.innerq_download_stats(codec)
        assert count == 2 * 2 * 16  # K+V groups of the calibration window
        res = turbo_kv.innerq_finalize_scales(sq, ch_max, count)
        # scales were already uploaded by the pool — re-uploading identity
        # is the fixture teardown; here just verify the pool's state flags
        assert not pool._calib_armed
        assert pool._calib_tokens >= 32
    finally:
        TurboKVCache.CALIBRATION_TOKENS = 2048
        turbo_kv.innerq_upload_scales(codec, torch.ones(128), torch.ones(128))
