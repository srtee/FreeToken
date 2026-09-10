"""GGUF reference dequant vs the vendored CUDA kernels.

``models/gguf/dequant.py`` carries a pure-torch reference for every block format
the engine can meet; the packed path never uses it (weights dequantize inside the
ggml kernels), so the reference is only as good as its agreement with the CUDA
ops. Each type here gets a random-but-valid block set (fp16 header fields drawn
from a sane range, all other bytes random -- every bit pattern is a legal code)
and the two paths must agree to fp16-storage rounding.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N_BLOCKS = 64


def _fp16_bytes(vals: torch.Tensor) -> torch.Tensor:
    return vals.to(torch.float16).view(torch.uint8)


def _scalar_block(ggml_type: int) -> torch.Tensor:
    """Random valid blocks for the 32-elem scalar formats (Q4_1, Q5_0, Q5_1, Q8_0)."""
    if ggml_type == 3:  # Q4_1: (d, min) + 16B nibbles
        head = _fp16_bytes(torch.randn(N_BLOCKS, 2).abs() * 0.1)
        return torch.cat([head, torch.randint(0, 256, (N_BLOCKS, 16), dtype=torch.uint8)], 1)
    if ggml_type in (6, 7):  # Q5_0: d + 4B qh + 16B nibbles; Q5_1: (d, min) + qh + nibbles
        head_n = 4 if ggml_type == 7 else 2
        head = _fp16_bytes(torch.randn(N_BLOCKS, head_n // 2).abs() * 0.1) if head_n >= 2 else None
        body = torch.cat(
            [
                torch.randint(0, 256, (N_BLOCKS, 4), dtype=torch.uint8),
                torch.randint(0, 256, (N_BLOCKS, 16), dtype=torch.uint8),
            ],
            1,
        )
        return torch.cat([head, body], 1) if head is not None else body
    assert ggml_type == 8  # Q8_0: d + 32 int8
    head = _fp16_bytes(torch.randn(N_BLOCKS, 1).abs() * 0.1)
    return torch.cat([head, torch.randint(0, 256, (N_BLOCKS, 32), dtype=torch.uint8)], 1)


def _k_block(ggml_type: int) -> torch.Tensor:
    """Random valid blocks for the 256-elem K-quants (Q2_K..Q6_K): random scale/quant
    bytes plus an fp16 super-block scale at the layout's tail/head position."""
    bytes_total = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210}[ggml_type]
    raw = torch.randint(0, 256, (N_BLOCKS, bytes_total), dtype=torch.uint8)
    d = _fp16_bytes(torch.randn(N_BLOCKS, 1).abs() * 0.1)
    if ggml_type == 10:  # scales(16) qs(64) dm(4): fp16 pair at 80
        raw[:, 80:84] = torch.cat([d, d], 1)
    elif ggml_type == 11:  # hmask(32) qs(64) scales(12) d(2): fp16 at 108
        raw[:, 108:110] = d
    elif ggml_type in (12, 13):  # dm(4) first: fp16 pair at 0
        raw[:, 0:4] = torch.cat([d, d], 1)
    else:  # Q6_K: ql(128) qh(64) scales(16) d(2): fp16 at 208
        raw[:, 208:210] = d
    return raw


@pytest.mark.parametrize(
    "ggml_type,block,rows",
    [
        (3, 32, _scalar_block),
        (6, 32, _scalar_block),
        (7, 32, _scalar_block),
        (8, 32, _scalar_block),
        (10, 256, _k_block),
        (11, 256, _k_block),
        (12, 256, _k_block),
        (13, 256, _k_block),
        (14, 256, _k_block),
    ],
    ids=["Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"],
)
def test_reference_dequant_matches_cuda_kernel(ggml_type: int, block: int, rows):
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.models.gguf.dequant import dequantize

    torch.manual_seed(ggml_type)
    raw = rows(ggml_type).cuda()

    ref = dequantize(raw, ggml_type, torch.float32).reshape(N_BLOCKS, block)
    ker = ggml_dequantize(raw, ggml_type, N_BLOCKS, block, torch.float32)

    assert torch.isfinite(ref).all(), "reference dequant produced non-finite values"
    assert torch.isfinite(ker).all(), "kernel dequant produced non-finite values"
    # Both paths dequantize through fp16 storage somewhere (scales / converted codes);
    # relative agreement at ~1e-3 of the value range is fp16-rounding noise.
    scale = ker.abs().max().clamp(min=1e-3)
    assert (ref - ker).abs().max() / scale < 2e-3