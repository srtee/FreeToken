"""Wave-0 oracle for the TurboQuant/TCQ KV codecs.

Pure-torch reference implementations of the buun-llama-cpp TurboQuant KV codecs,
kept bit-compatible with the CUDA kernels:

- ``turbo_rotate``: FWHT-128 with the fixed seed-42 sign vectors
  (d_turbo_wht_s1/s2), normalized by 1/sqrt(128). Inverse is the same transform
  (signs1/s2 swap) — FWHT is an involution up to the sign swap.
- ``turbo4`` / ``turbo8``: scalar quantizers against the compiled-in
  Lloyd-Max / uniform tables, matching ``quantize_f32_turbo4_0_block`` /
  ``quantize_f32_turbo8_0_block`` byte-for-byte (block_turbo4_0 = 66 B,
  block_turbo8_0 = 130 B per 128-value group).
- ``turbo3_tcq`` / ``turbo2_tcq``: right-shift trellis Viterbi encoders over the
  trained codebooks loaded from ``codebooks/`` .bin files, producing the
  390/262-bit bitstreams (block_turbo3_tcq = 52 B, block_turbo2_tcq = 36 B).
  Decode reads the state at t*bits from the packed stream: O(1) per element.

Bitstream layout (encoder and decoder must agree exactly):
  qs[0..] = 6-bit initial-state prefix (state >> L_shift, L=3 or 2) followed by
  128 quantized output symbols packed LSB-first. Decode of token t:
    state_t = read_bits(qs, 6 + t*bits, bits)  (t >= L_shift*... see below)
  where state_t for t < k_bits is reconstructed by shifting in the initial
  prefix. Recon value = codebook[state_t] * corrected_norm, corrected_norm =
  ||x|| / ||codebook path|| * alpha.

Block-struct notes (ggml-common.h, byte-exact):
  block_turbo3_tcq: norm fp16 (2B) + qs[49] + pad(1B)  = 52 B
  block_turbo2_tcq: norm fp16 (2B) + qs[33] + pad(1B)  = 36 B
  block_turbo4_0:   norm fp16 (2B) + qs[64]            = 66 B
  block_turbo8_0:   norm fp16 (2B) + qs[128]           = 130 B

Usage (see scripts/tcq_oracle.py for the full KLD harness):
    from freetoken.kernel.turbo_oracle import TurboCodec, turbo_rotate
    codec = TurboCodec("turbo3_tcq")
    packed = codec.encode(x)   # (N, 52) uint8, N groups of 128
    recon  = codec.decode(packed, alpha=1.0)
"""

from __future__ import annotations

import functools
import math
import pathlib
import struct

import torch


# Lloyd-Max / uniform codebooks, copied verbatim from buun
# turbo-quant-cuda.cuh (d_turbo_centroids_4bit / _8bit and the mid tables).
TURBO4_CENTROIDS = (
    -0.241556, -0.182907, -0.143047, -0.111065,
    -0.083317, -0.058069, -0.034311, -0.011353,
    0.011353, 0.034311, 0.058069, 0.083317,
    0.111065, 0.143047, 0.182907, 0.241556,
)
TURBO4_MIDS = (
    -0.212232, -0.162977, -0.127056, -0.097191, -0.070693,
    -0.046190, -0.022832, 0.000000, 0.022832, 0.046190,
    0.070693, 0.097191, 0.127056, 0.162977, 0.212232,
)
# turbo8: uniform grid centroid[i] = (i - 127.5) / 127.5 in [-1, 1]; per-block
# absmax scale is stored in the norm slot (norm = ||x|| * absmax).
TURBO8_CENTROIDS = tuple((i - 127.5) / 127.5 for i in range(256))

# Decode-side alpha_V constants (turbo-tcq-alpha.cuh) — flat optima for the
# coord-descent codebooks. K stays at 1.0 (softmax-neutral).
TCQ_ALPHA_V = {"turbo3_tcq": 1.02, "turbo2_tcq": 1.06}

INV_SQRT_128 = 0.08838834764831845


def _wht_signs(seed: int) -> torch.Tensor:
    """d_turbo_wht_s1/s2 sign vectors from turbo-wht.cu (seed 42)."""
    if seed != 42:
        raise ValueError(f"no sign table for seed {seed}")
    s1 = [
        -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
        1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
        -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
        1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
        -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1,
        1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1,
        -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
        1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
    ]
    s2 = [
        1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1,
        1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
        1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,
        1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
        1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
        -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
        1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
        -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
    ]
    return torch.tensor(s1, dtype=torch.float32), torch.tensor(
        s2, dtype=torch.float32
    )


_S1, _S2 = _wht_signs(42)


def _fwht_matrix() -> torch.Tensor:
    """128x128 unnormalized Hadamard."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < 128:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    assert h.shape == (128, 128)
    return h


_H = _fwht_matrix()


def turbo_rotate(x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """FWHT-128 rotation with the seed-42 sign vectors on the last dim.

    Forward: x * s1 -> FWHT / sqrt(128) -> * s2. Inverse swaps s1/s2 (the CUDA
    dequant path does the same via direction=1).
    """
    s1, s2 = (_S2, _S1) if inverse else (_S1, _S2)
    return x * s1 @ _H * INV_SQRT_128 * s2




def _unpack_bytes_to_bits(qs: torch.Tensor) -> torch.Tensor:
    """(N, B) uint8 -> (N, B*8) uint8, LSB-first per byte (matches the CUDA writer)."""
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint16)
    bits = ((qs.to(torch.uint16)[..., None] & weights) != 0).to(torch.uint8)
    return bits.reshape(*qs.shape[:-1], qs.shape[-1] * 8)


def _read_bits(bits_row: torch.Tensor, pos: int, width: int) -> int:
    v = 0
    for b in range(width):
        v |= int(bits_row[pos + b]) << b
    return v


def _viterbi_encode(
    x: torch.Tensor, codebook: torch.Tensor, bits: int, n_states: int
) -> tuple[int, torch.Tensor]:
    """Right-shift trellis Viterbi over one normalized 128-group, matching the
    buun trellis contract exactly (scripts/tcq_rshift.py RightShiftTrellis):

      next_state(s, out) = (s >> bits) | (out << 6)      # L=6+bits
      decode: state_t = read_bits(qs, t*bits, 6+bits)    # sliding window

    The reference trainer runs the forward pass FROM STATE 0 (free-init
    variants start from all-equal costs) reading x[0..127], stores the per-step
    path states, and the CUDA bitstream stores 6 prefix bits + the per-step
    output symbols (state >> 6). The sliding-window read reproduces the path
    states exactly because state_t's low 6 bits are state_{t-1}'s top 6 bits,
    which are the previous symbols' bits in the packed stream.

    Returns (initial_state, outputs[128]) — `initial_state` is the state the
    forward pass started from (0 for the fixed-init trellis; its low `bits`
    bits are the stream prefix).
    """
    T = x.shape[0]
    n_out = 1 << bits
    shift = bits
    cb = codebook.to(torch.float32)
    INF = 3.4028234663852886e38
    # Forward from state 0 (free-init semantics = cost 0 everywhere would let
    # the argmin pick any start; buun's shipped encoders pin state 0).
    # Accumulation is fp32 to be bit-compatible with the CUDA encoder (the
    # trellis costs accumulate over 128 steps; fp64 oracle costs flip near-ties).
    cost = torch.full((n_states,), INF, dtype=torch.float32)
    cost[0] = 0.0
    bt_prev = torch.zeros(T, n_states, dtype=torch.int32)
    for t in range(T):
        new_cost = torch.full((n_states,), INF, dtype=torch.float32)
        xt = float(x[t])
        # Target-side min, mirroring the CUDA encoder exactly: for each target
        # state s, cost[s] = min over predecessors p of cost_rd[(s & 0x3F) << bits | p]
        # + (xt - cb[s])^2. Every state's cost is written every step (INF
        # propagates through unreachable predecessors), so no finite-mask.
        g = torch.arange(n_states) & 0x3F
        base_prev = g << bits
        preds = base_prev.unsqueeze(1) + torch.arange(1 << bits)  # (n_states, n_out)
        best = preds.shape[1] and torch.min(cost[preds], dim=1).values  # (n_states,)
        dist = (xt - cb) ** 2
        new_cost = best + dist
        # backpointer: argmin predecessor per target
        best_p = torch.argmin(cost[preds], dim=1).to(torch.int32)
        bt_prev[t] = ((g << bits) | best_p)
        cost = new_cost
    final = int(cost.argmin())
    states = torch.zeros(T, dtype=torch.int32)
    state = final
    for t in range(T - 1, -1, -1):
        states[t] = state
        state = int(bt_prev[t, state])
    initial = state
    assert initial == 0, f"forward from state 0 must backtrack to 0, got {initial}"
    outputs = (states >> 6).to(torch.int32)
    # sliding-window decode check (read_9_bits(qs, t*bits) == states[t])
    st = 0
    for t in range(T):
        st = (st >> shift) | (int(outputs[t]) << 6)
        assert st == int(states[t]), (t, st, int(states[t]))
    return initial, outputs


def _viterbi_recon_norm(
    initial: int, outputs: torch.Tensor, codebook: torch.Tensor, bits: int
) -> float:
    """Reconstruction norm of the chosen path (for corrected_norm). Mirrors the
    CUDA recon: s_t = (s_{t-1} >> bits) | (out_t << 6) starting from the stored
    initial state."""
    state = initial
    sq = 0.0
    cb = codebook.to(torch.float64)
    for t in range(outputs.numel()):
        sq += float(cb[state]) ** 2
        state = (state >> bits) | (int(outputs[t]) << 6)
    return math.sqrt(sq)

CODEBOOK_PACKAGE_DIR = pathlib.Path(__file__).parent / "codebooks"

# Compiled-in defaults: buun's coord-descent split books (K and V differ — the
# V book absorbs the alpha_V boost at training time). Extracted verbatim from
# turbo-quant-cuda.cuh's d_turbo*_tcq_codebook[_v] constants so the torch
# oracle, the CUDA kernels, and any dump tooling agree on one table.
DEFAULT_TCQ_CODEBOOKS = {
    "turbo3_tcq": ("tcq3bit_k.bin", "tcq3bit_v.bin", 512),
    "turbo2_tcq": ("tcq2bit_k.bin", "tcq2bit_v.bin", 256),
}


def _load_codebook(name: str, is_v: bool) -> torch.Tensor:
    k_rel, v_rel, n = DEFAULT_TCQ_CODEBOOKS[name]
    path = CODEBOOK_PACKAGE_DIR / (v_rel if is_v else k_rel)
    data = path.read_bytes()
    assert len(data) == n * 4, f"{path}: expected {n} f32, got {len(data)} B"
    return torch.tensor(struct.unpack(f"{n}f", data), dtype=torch.float32)


@functools.lru_cache(maxsize=None)
def _tcq_codebook(name: str, is_v: bool = False) -> torch.Tensor:
    return _load_codebook(name, is_v)


class TurboCodec:
    """One codec, one 128-value group at a time. Storage dtype is uint8 rows."""

    # name -> (block_bytes, bits, n_states, is_tcq)
    SPECS = {
        "turbo4": (66, 4, 0, False),
        "turbo8": (130, 8, 0, False),
        "turbo3_tcq": (52, 3, 512, True),
        "turbo2_tcq": (36, 2, 256, True),
    }


    def __init__(
        self,
        name: str,
        is_v: bool = False,
        codebook_override: torch.Tensor | None = None,
    ):
        if name not in self.SPECS:
            raise ValueError(f"unknown codec {name!r}")
        self.name = name
        self.is_v = is_v
        self.block_bytes, self.bits, self.n_states, self.is_tcq = self.SPECS[name]
        if self.is_tcq:
            self.codebook = (
                codebook_override
                if codebook_override is not None
                else _tcq_codebook(name, is_v)
            ).clone()
            assert self.codebook.numel() == self.n_states
        else:
            self.codebook = None

    # ---- static (turbo4/turbo8) -------------------------------------------------

    def _quant_turbo4(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 128) -> (N, 66) matching block_turbo4_0: norm fp16 + packed nibbles
        (low nibble first, low nibble at even j). Pipeline mirrors buun's
        quantize_f32_turbo4_0_block: L2-normalize -> FWHT(signs1, signs2) ->
        4-bit Lloyd-Max quantize in the rotated domain."""
        mids = torch.tensor(TURBO4_MIDS, dtype=x.dtype, device=x.device)
        cent = torch.tensor(TURBO4_CENTROIDS, dtype=x.dtype, device=x.device)
        norm = (x**2).sum(dim=-1, keepdim=True).sqrt()
        xn = torch.where(norm > 1e-10, x / norm, x)
        xr = turbo_rotate(xn)
        idx = torch.bucketize(xr, mids)  # right=False: mids[i-1] <= x < mids[i] -> idx i
        recon = cent[idx]
        recon_norm = (recon**2).sum(dim=-1, keepdim=True).sqrt()
        corrected = torch.where(recon_norm > 1e-10, norm / recon_norm, norm)
        norm_bytes = corrected.half().view(torch.uint8)  # (N, 2)
        # pack nibbles: qs[j/2] = (idx[j+1] << 4) | idx[j]
        nib = idx.to(torch.uint8)
        qs = nib[:, 0::2] | (nib[:, 1::2] << 4)
        return torch.cat([norm_bytes, qs], dim=-1)

    def _dequant_turbo4(self, blocks: torch.Tensor) -> torch.Tensor:
        norm = blocks[:, :2].contiguous().view(torch.float16).float()  # (N, 1)
        qs = blocks[:, 2:]
        lo = (qs & 0xF).float()
        hi = (qs >> 4).float()
        idx = torch.stack([lo, hi], dim=-1).reshape(blocks.shape[0], 128).long()
        cent = torch.tensor(TURBO4_CENTROIDS, dtype=torch.float32)
        return cent[idx] * norm

    def _quant_turbo8(self, x: torch.Tensor) -> torch.Tensor:
        norm = (x**2).sum(dim=-1, keepdim=True).sqrt()
        xn = torch.where(norm > 1e-10, x / norm, x)
        xr = turbo_rotate(xn)
        absmax = xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        inv = 1.0 / absmax
        # lrintf round-half-to-even at .5? CUDA lrintf uses the current rounding
        # mode (round-nearest-even), matching torch.round's half-to-even only at
        # exact .5 boundaries; the grid hits .5 boundaries at x*inv = k+0.5 over
        # 255 levels — measure MSE, don't chase bit-equality here.
        idx = torch.round(xr * inv * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        # norm slot stores ||x|| * absmax (buun: dst->norm = __float2half(norm * scale))
        stored = (norm * absmax).half().view(torch.uint8)
        return torch.cat([stored, idx], dim=-1)

    def _dequant_turbo8(self, blocks: torch.Tensor) -> torch.Tensor:
        norm = blocks[:, :2].contiguous().view(torch.float16).float()
        qs = blocks[:, 2:].long()
        cent = torch.tensor(TURBO8_CENTROIDS, dtype=torch.float32)
        return cent[qs] * norm

    # ---- TCQ -------------------------------------------------------------------

    def _quant_tcq(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """(N, 128) -> (N, block_bytes). One Viterbi per row (reference speed:
        python loop, fine for a few thousand groups)."""
        N = x.shape[0]
        out = torch.zeros(N, self.block_bytes, dtype=torch.uint8)
        cb = self.codebook.to(torch.float32)
        for r in range(N):
            norm = x[r].float().norm()
            xn = x[r].float() / norm if norm > 1e-10 else x[r].float()
            # buun's TCQ encoders run the FWHT rotation before the Viterbi;
            # decode is O(1) in the ROTATED domain (the fused FA path and the
            # materializer both consume rotated values; un-rotation happens at
            # the attention-output stage via the inverse transform).
            xn = turbo_rotate(xn.unsqueeze(0))[0]
            initial, outputs = _viterbi_encode(xn, cb, self.bits, self.n_states)
            recon_norm = _viterbi_recon_norm(initial, outputs, cb, self.bits)
            corrected = (
                (norm / recon_norm) if recon_norm > 1e-10 else norm
            ) * (alpha if alpha != 1.0 else 1.0)
            out[r, :2] = (torch.tensor([corrected], dtype=torch.float16).view(torch.uint8))
            # The decoder reads states at bit offset t*bits (NO 6-bit gap): the
            # 6-bit prefix is what makes the first `bits` reads see the initial
            # state's low bits, and the window slides from there. So the stream
            # is: prefix bits at [0,6), then symbols packed at [6, 6+T*bits) —
            # and read_bits(t*bits) covers prefix tail + symbols, reproducing
            # state_t. For initial state 0 the prefix is 6 zero bits.
            prefix = initial & 0x3F if self.n_states <= 512 else 0
            # NOTE: with initial=0 the prefix is zeros; for the general free-init
            # case the prefix is the LOW 6 bits of the initial state... but the
            # sliding window reads at t*bits starting at 0, so the first 9-bit
            # window [0,9) = prefix(6) + sym0(3) = state_0 only if state_0's low
            # 6 bits are the prefix and its high 3 bits are sym0. state_0 =
            # (initial >> 3) | (out_0 << 6)?? The pinned init-0 trainer never
            # needs a nonzero prefix: keep 6 zero bits and assert initial==0.
            assert initial == 0
            prefix = 0
            bits_row = torch.zeros(6 + outputs.numel() * self.bits, dtype=torch.uint8)
            for b in range(6):
                bits_row[b] = (prefix >> b) & 1
            for i in range(outputs.numel()):
                sym = int(outputs[i])
                for b in range(self.bits):
                    if (sym >> b) & 1:
                        bits_row[6 + i * self.bits + b] = 1
            # qs is (block_bytes - 3) bytes: norm(2) + pad(1) removed. The last
            # byte's tail bits (2 for both codecs) are zero padding.
            n_bytes = self.block_bytes - 3
            padded = torch.cat([bits_row, torch.zeros(n_bytes * 8 - bits_row.numel(), dtype=torch.uint8)])
            weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
            out[r, 2 : 2 + n_bytes] = (padded.reshape(n_bytes, 8) * weights).sum(dim=1).to(torch.uint8)
        return out

    def _dequant_tcq(self, blocks: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """(N, block_bytes) -> (N, 128) float. O(1) per element: sliding-window
        read of the full (6+bits)-bit state at bit offset t*bits, gather
        codebook[state]. The stored norm already carries alpha (encode
        multiplied it in), so decode uses it directly."""
        norm = blocks[:, :2].contiguous().view(torch.float16).float()  # (N, 1)
        qs = blocks[:, 2 : self.block_bytes - 1]
        bits_row = _unpack_bytes_to_bits(qs)  # (N, nbytes*8)
        n_bytes = self.block_bytes - 3
        bits_row = bits_row[:, : n_bytes * 8]
        states = torch.zeros(blocks.shape[0], 128, dtype=torch.long)
        for t in range(128):
            pos = t * self.bits
            v = torch.zeros(blocks.shape[0], dtype=torch.long)
            for b in range(6 + self.bits):
                v |= bits_row[:, pos + b].long() << b
            states[:, t] = v
        recon = self.codebook[states].float()
        return recon * norm
    def encode(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """(N, 128) float -> (N, block_bytes) uint8."""
        assert x.shape[-1] == 128
        x = x.float().reshape(-1, 128)
        if self.name == "turbo4":
            return self._quant_turbo4(x)
        if self.name == "turbo8":
            return self._quant_turbo8(x)
        return self._quant_tcq(x, alpha)

    def decode(self, blocks: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        """(N, block_bytes) uint8 -> (N, 128) float."""
        if self.name == "turbo4":
            return self._dequant_turbo4(blocks)
        if self.name == "turbo8":
            return self._dequant_turbo8(blocks)
        assert alpha is None, "alpha is baked into the stored norm at encode"
        return self._dequant_tcq(blocks)

    def roundtrip(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.decode(self.encode(x, alpha=alpha))