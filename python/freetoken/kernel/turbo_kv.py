"""TurboQuant/TCQ KV cache kernels — JIT loader and python entry points.

Follows ``store.py``'s pattern: ``KernelConfig`` + ``load_jit`` over
``csrc/jit/turbo_kv.cu``, with the TCQ codebooks uploaded once per module load
(extracted from buun's compiled-in constants into ``codebooks/*.bin``).

Block layouts (byte-identical with buun ggml-common.h):
  turbo4:     66 B  = 2 B fp16 norm + 64 B nibble-packed 4-bit indices
  turbo8:     130 B = 2 B fp16 norm + 128 B 8-bit indices
  turbo3_tcq: 52 B  = 2 B fp16 norm + 49 B bitstream + 1 B pad
  turbo2_tcq: 36 B  = 2 B fp16 norm + 33 B bitstream + 1 B pad

One block = one 128-value rotation group (head_dim=128). K blocks use the K
codebook, V blocks the V codebook (TCQ split books).
"""

from __future__ import annotations

import functools
import pathlib
from typing import TYPE_CHECKING

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

CODEBOOK_DIR = pathlib.Path(__file__).parent / "codebooks"

# codec name -> (kernel codec id, block_bytes)
CODEC_SPECS: dict[str, tuple[int, int]] = {
    "turbo4": (4, 66),
    "turbo8": (8, 130),
    "turbo3_tcq": (32, 52),
    "turbo2_tcq": (22, 36),
}

TCQ_CODEBOOK_FILES = {
    "turbo3_tcq": ("tcq3bit_k.bin", "tcq3bit_v.bin"),
    "turbo2_tcq": ("tcq2bit_k.bin", "tcq2bit_v.bin"),
}


def block_bytes(codec: str) -> int:
    return CODEC_SPECS[codec][1]


def _load_codebook(path: pathlib.Path, n: int) -> torch.Tensor:
    data = path.read_bytes()
    assert len(data) == n * 4, f"{path}: expected {n} f32, got {len(data)} B"
    return torch.frombuffer(bytearray(data), dtype=torch.float32).clone()


@functools.cache
def _jit_turbo_module(codec_id: int, block_bytes: int) -> Module:
    args = make_cpp_args(codec_id, block_bytes)
    return load_jit(
        "turbo_kv",
        *args,
        cuda_files=["turbo_kv.cu"],
        cuda_wrappers=[
            ("launch", f"TurboQuantLaunch<{args}>::run"),
            ("upload", "TurboCodebookUpload::run"),
        ],
    )


@functools.cache
def _jit_dequant_module() -> Module:
    return load_jit(
        "turbo_kv_dequant",
        cuda_files=["turbo_kv.cu"],
        cuda_wrappers=[
            ("dequant", "TurboDequantLaunch::run"),
            ("upload", "TurboCodebookUpload::run"),
        ],
    )


def _upload_into(module: Module) -> None:
    """Upload the TCQ codebooks into THIS module's __device__ arrays (each
    JIT module carries its own copy of the symbols)."""
    cb3k = _load_codebook(CODEBOOK_DIR / "tcq3bit_k.bin", 512)
    cb3v = _load_codebook(CODEBOOK_DIR / "tcq3bit_v.bin", 512)
    cb2k = _load_codebook(CODEBOOK_DIR / "tcq2bit_k.bin", 256)
    cb2v = _load_codebook(CODEBOOK_DIR / "tcq2bit_v.bin", 256)
    module.upload(cb3k, cb3v, cb2k, cb2v)


@functools.cache
def _upload_quant_module(codec_id: int, block_bytes: int) -> Module:
    module = _jit_turbo_module(codec_id, block_bytes)
    _upload_into(module)
    return module


@functools.cache
def _upload_dequant_module() -> Module:
    module = _jit_dequant_module()
    _upload_into(module)
    return module


def upload_codebooks() -> None:
    """Upload the TCQ codebooks to every live module (idempotent). Kept for the
    eager path; the quant/dequant entry points upload lazily via the cached
    wrappers above."""
    _upload_dequant_module()




def turbo_quantize(
    codec: str,
    src: torch.Tensor,
    dst: torch.Tensor,
    locs: torch.Tensor,
    is_v: bool,
) -> None:
    """Quantize ``src`` (L, heads, 128) into ``dst`` (L, heads, block_bytes) at
    rows ``locs`` of the caller's slab — wait, dst here IS (L, heads, block_bytes)
    preallocated by the pool; the pool scatters it into the slab via the existing
    index kernel, or passes slab + out_loc directly. See TurboKVCache.store_kv."""
    codec_id, bb = CODEC_SPECS[codec]
    module = _upload_quant_module(codec_id, bb)
    module.launch(src, dst, locs, int(is_v))


def turbo_dequantize(
    codec: str,
    src: torch.Tensor,
    dst: torch.Tensor,
    locs: torch.Tensor,
    is_v: bool,
) -> None:
    """Dequantize rows ``locs`` of slab ``src`` (tokens, heads, block_bytes) into
    ``dst`` (L, heads, 128) fp16."""
    module = _upload_dequant_module()
    module.dequant(src, dst, locs, CODEC_SPECS[codec][0], int(is_v))