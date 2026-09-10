"""GGML block-quant dequantization in pure torch.

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c`` (scalar blocks: the ``dequantize_*.cuh`` device
functions; K-quants: the ``dequantize_row_*`` CUDA kernels in ``dequantize.cuh``).
"""
from __future__ import annotations


import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type. Mirrors gguf-py's
# ``GGML_QUANT_SIZES`` for every type the vendored kernels dispatch (the iq* row
# below is dequant-only: MMVQ/MMQ exist, but no torch reference is shipped --
# tests cross-check those through the CUDA kernel alone).
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    # iq* (dequant-only coverage; shapes from gguf-py GGML_QUANT_SIZES).
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    29: (256, 56),   # IQ1_M
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
}



def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q4_1(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_1: per 32-elem block = fp16 (d, min) + 16 packed nibbles; ``w = d*q + min``."""
    raw = raw.reshape(-1, 20)
    dm = raw[:, 0:4].contiguous().view(torch.float16).to(torch.float32)  # [N,2]
    qs = raw[:, 4:20]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return (q * dm[:, :1] + dm[:, 1:]).reshape(-1).to(out_dtype)


def dequant_q5_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_0: per 32-elem block = fp16 scale + 4B high bits + 16 packed nibbles;
    ``w = d*(q5 - 16)``. Storage order mirrors Q4_0's lo/hi nibble split."""
    raw = raw.reshape(-1, 22)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    b = raw[:, 2:6].to(torch.int64)  # [N,4] little-endian u32 bytes
    qh = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)  # [N]
    qs = raw[:, 6:22]  # [N,16]
    lo = (qs & 0x0F).to(torch.int32)
    hi = (qs >> 4).to(torch.int32)
    # cuh: xh_0 = ((qh >> (iqs+0)) << 4) & 0x10 for the low nibble of byte iqs;
    # xh_1 = (qh >> (iqs+12)) & 0x10 for the high nibble.
    idx = torch.arange(16, device=raw.device)
    xh_0 = ((qh.unsqueeze(1) >> idx) << 4) & 0x10
    xh_1 = (qh.unsqueeze(1) >> (idx + 12)) & 0x10
    q = torch.cat([lo | xh_0, hi | xh_1], dim=1).to(torch.float32)  # [N,32]
    return ((q - 16.0) * d).reshape(-1).to(out_dtype)


def dequant_q5_1(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_1: like Q5_0 but ``w = d*q5 + min`` with an fp16 (d, min) header."""
    raw = raw.reshape(-1, 24)
    dm = raw[:, 0:4].contiguous().view(torch.float16).to(torch.float32)  # [N,2]
    b = raw[:, 4:8].to(torch.int64)  # [N,4] little-endian u32 bytes
    qh = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)  # [N]
    qs = raw[:, 8:24]  # [N,16]
    lo = (qs & 0x0F).to(torch.int32)
    hi = (qs >> 4).to(torch.int32)
    idx = torch.arange(16, device=raw.device)
    xh_0 = ((qh.unsqueeze(1) >> idx) << 4) & 0x10
    xh_1 = (qh.unsqueeze(1) >> (idx + 12)) & 0x10
    q = torch.cat([lo | xh_0, hi | xh_1], dim=1).to(torch.float32)  # [N,32]
    return (q * dm[:, :1] + dm[:, 1:]).reshape(-1).to(out_dtype)


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q8_0: per 32-elem block = fp16 scale + 32 int8 quants; ``w = d*q``."""
    raw = raw.reshape(-1, 34)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    q = raw[:, 2:34].view(torch.int8).to(torch.float32)  # [N,32]
    return (q * d).reshape(-1).to(out_dtype)


def _scale_min_k4(j: int, sc: torch.Tensor):
    """Vectorized ``get_scale_min_k4`` (dequantize.cuh): j in 0..7 over the 12-byte
    6-bit scale+min table of a K-superblock; returns (scale, min) as [N] int tensors."""
    if j < 4:
        d = sc[:, j] & 63
        m = sc[:, j + 4] & 63
    else:
        d = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
        m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
    return d.to(torch.int32), m.to(torch.int32)


def dequant_q2_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q2_K: 256-elem super-block = 16B 4-bit scale/min pairs + 64B 2-bit quants +
    fp16 (d, dmin). Per 16-byte scale chunk ``is``: low nibble = scale, high = min;
    each byte of quants packs four 2-bit codes (sub-blocks of 32)."""
    raw = raw.reshape(-1, 84)
    n = raw.shape[0]
    sc = raw[:, 0:16].to(torch.int32)  # [n,16]
    qs = raw[:, 16:80]  # [n,64]
    dm = raw[:, 80:84].contiguous().view(torch.float16).to(torch.float32)  # [n,2]
    dall, dmin = dm[:, :1], dm[:, 1:]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # CUDA layout (64 threads/block): tid -> n = tid/32 in {0,1}, l = tid%32 in 0..31;
    # q byte = qs[32*n + l]; y base = 128*n; y[l + 32*shift] uses scales[8n + l//16 + 2*shift]
    # (low nibble = scale, high nibble = min).
    for n2 in range(2):  # two 128-elem halves
        q = qs[:, 32 * n2:32 * n2 + 32]  # [n,32] bytes
        for shift in range(4):  # 2-bit field within the byte
            for half in range(2):  # l<16 vs l>=16 (scale byte offset +0 / +1)
                is_idx = 8 * n2 + half + 2 * shift
                sc_lo = (sc[:, is_idx] & 0xF).to(torch.float32).unsqueeze(1)  # [n,1]
                mn_lo = (sc[:, is_idx] >> 4).to(torch.float32).unsqueeze(1)
                code = ((q[:, 16 * half:16 * half + 16] >> (2 * shift)) & 3).to(torch.float32)
                y[:, 128 * n2 + 32 * shift + 16 * half:128 * n2 + 32 * shift + 16 * half + 16] = (
                    dall * sc_lo * code - dmin * mn_lo
                )
    return y.reshape(-1).to(out_dtype)


def dequant_q3_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q3_K: 256-elem super-block = 32B high-bit mask + 64B low 2-bit quants +
    12B 6-bit scales + fp16 d. Mirrors the CUDA ``dequantize_block_q3_K`` thread
    decomposition: thread r -> (n, j, is0), writing 32-elem strip y[32*j + l0..]."""
    raw = raw.reshape(-1, 110)
    n = raw.shape[0]
    hmask = raw[:, 0:32]  # [n,32]
    qs = raw[:, 32:96]  # [n,64]
    sc = raw[:, 96:108].to(torch.int32)  # [n,12]
    d = _f16_scales(raw, 108, 110)  # [n,1]
    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # CUDA (64 threads): r = tid/4 (0..15), n = (r/2)/4 in {0,1}, j = (r/2)%4,
    # is0 = r%2, l0 = 16*is0 + 4*(tid%4). Per (n, j, is0, t4=tid%4): writes
    # y[128n + 32j + l0 .. +3] from q[l]>>shift, sign from hmask[l] bit (4n+j).
    for nq in range(2):
        q = qs[:, 32 * nq:32 * nq + 32]  # [n,32] bytes for this 128-elem strip
        for j in range(4):
            m = 1 << (4 * nq + j)
            for is0 in range(2):
                is_idx = 8 * nq + 2 * j + is0
                if is_idx < 4:
                    us = (sc[:, is_idx:is_idx + 1] & 0xF) | (((sc[:, is_idx + 8:is_idx + 9] >> 0) & 3) << 4)
                elif is_idx < 8:
                    us = (sc[:, is_idx:is_idx + 1] & 0xF) | (((sc[:, is_idx + 4:is_idx + 5] >> 2) & 3) << 4)
                elif is_idx < 12:
                    us = (sc[:, is_idx - 8:is_idx - 7] >> 4) | (((sc[:, is_idx:is_idx + 1] >> 4) & 3) << 4)
                else:
                    us = (sc[:, is_idx - 8:is_idx - 7] >> 4) | (((sc[:, is_idx - 4:is_idx - 3] >> 6) & 3) << 4)
                dl = d * (us - 32).to(torch.float32)  # [n,1]
                shift = 2 * j
                for t4 in range(4):
                    l0 = 16 * is0 + 4 * t4
                    for l in range(l0, l0 + 4):
                        code = ((q[:, l] >> shift) & 3).to(torch.float32)  # [n]
                        sign = torch.where((hmask[:, l] & m) != 0, 0.0, 4.0)  # [n]
                        y[:, 128 * nq + 32 * j + l] = dl.squeeze(1) * (code - sign)
    return y.reshape(-1).to(out_dtype)


def dequant_q4_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_K: 256-elem super-block = fp16 (d, dmin) + 12B 6-bit scales + 128B nibbles.
    Mirror of the CUDA ``dequantize_block_q4_K``: strip il writes y[64*il + 4*ir]
    (low nibble, scale is=2*il) and y[+32] (high nibble, scale is=2*il+1)."""
    raw = raw.reshape(-1, 144)
    n = raw.shape[0]
    dm = raw[:, 0:4].contiguous().view(torch.float16).to(torch.float32)  # [n,2]
    sc = raw[:, 4:16].to(torch.int32)  # [n,12]
    qs = raw[:, 16:144]  # [n,128]
    dall, dmin = dm[:, :1], dm[:, 1:]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    for il in range(4):  # 64-elem strip
        q = qs[:, 32 * il:32 * il + 32]  # [n,32] nibble bytes
        s0, m0 = _scale_min_k4(2 * il, sc)
        s1, m1 = _scale_min_k4(2 * il + 1, sc)
        d1 = dall * s0.unsqueeze(1).to(torch.float32)
        m1v = dmin * m0.unsqueeze(1).to(torch.float32)
        d2 = dall * s1.unsqueeze(1).to(torch.float32)
        m2v = dmin * m1.unsqueeze(1).to(torch.float32)
        base = 64 * il
        lo = (q & 0xF).to(torch.float32)
        hi = (q >> 4).to(torch.float32)
        y[:, base + 0:base + 32] = d1 * lo - m1v
        y[:, base + 32:base + 64] = d2 * hi - m2v
    return y.reshape(-1).to(out_dtype)


def dequant_q5_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q5_K: like Q4_K plus a 32B high-bit table (5th bit of each nibble code)."""
    raw = raw.reshape(-1, 176)
    n = raw.shape[0]
    dm = raw[:, 0:4].contiguous().view(torch.float16).to(torch.float32)  # [n,2]
    sc = raw[:, 4:16].to(torch.int32)  # [n,12]
    qh = raw[:, 16:48]  # [n,32] high bits
    qs = raw[:, 48:176]  # [n,128]
    dall, dmin = dm[:, :1], dm[:, 1:]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # CUDA ir loop: ql = qs + 32*il + 2*ir, qh = qh + 2*ir; y[64il+2ir], y[64il+2ir+1]
    # from ql[0], ql[1] low nibbles with qh[0], qh[1] bit hm; y[64il+32+...] the high
    # nibbles with bit hm<<1. Vectorized over ir: even/odd byte columns.
    for il in range(4):
        ql = qs[:, 32 * il:32 * il + 32]  # [n,32]
        qh0, qh1 = qh[:, 0:32:2], qh[:, 1:32:2]  # [n,16] each (indexed by ir, not il)
        s0, m0 = _scale_min_k4(2 * il, sc)
        s1, m1 = _scale_min_k4(2 * il + 1, sc)
        d1 = dall * s0.unsqueeze(1).to(torch.float32)  # [n,1]
        m1v = dmin * m0.unsqueeze(1).to(torch.float32)
        d2 = dall * s1.unsqueeze(1).to(torch.float32)
        m2v = dmin * m1.unsqueeze(1).to(torch.float32)
        hm = 1 << (2 * il)
        base = 64 * il
        ql0, ql1 = ql[:, 0::2], ql[:, 1::2]  # [n,16] each
        # qh0/qh1 already sliced above (ir-indexed)
        lo0 = (ql0 & 0xF).to(torch.float32) + 16.0 * ((qh0 & hm) != 0).to(torch.float32)
        lo1 = (ql1 & 0xF).to(torch.float32) + 16.0 * ((qh1 & hm) != 0).to(torch.float32)
        hi0 = (ql0 >> 4).to(torch.float32) + 16.0 * ((qh0 & (hm << 1)) != 0).to(torch.float32)
        hi1 = (ql1 >> 4).to(torch.float32) + 16.0 * ((qh1 & (hm << 1)) != 0).to(torch.float32)
        y[:, base + 0:base + 32:2] = d1 * lo0 - m1v
        y[:, base + 1:base + 32:2] = d1 * lo1 - m1v
        y[:, base + 32:base + 64:2] = d2 * hi0 - m2v
        y[:, base + 33:base + 64:2] = d2 * hi1 - m2v
    return y.reshape(-1).to(out_dtype)




def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q4_1: dequant_q4_1,
    GGML_Q5_0: dequant_q5_0,
    GGML_Q5_1: dequant_q5_1,
    GGML_Q8_0: dequant_q8_0,
    GGML_Q2_K: dequant_q2_k,
    GGML_Q3_K: dequant_q3_k,
    GGML_Q4_K: dequant_q4_k,
    GGML_Q5_K: dequant_q5_k,
    GGML_Q6_K: dequant_q6_k,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q4_1",
    "GGML_Q5_0",
    "GGML_Q5_1",
    "GGML_Q8_0",
    "GGML_Q2_K",
    "GGML_Q3_K",
    "GGML_Q4_K",
    "GGML_Q5_K",
    "GGML_Q6_K",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q4_1",
    "dequant_q5_0",
    "dequant_q5_1",
    "dequant_q8_0",
    "dequant_q2_k",
    "dequant_q3_k",
    "dequant_q4_k",
    "dequant_q5_k",
    "dequant_q6_k",
    "dequantize",
]
