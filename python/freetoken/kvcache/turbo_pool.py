"""TurboQuant/TCQ KV pool — quantized storage with a dequantizing materializer.

Storage layout: one uint8 slab per tensor (K and V), shaped
``(num_storage_layers, tokens, local_kv_heads, block_bytes)`` — the same 6-D
logical geometry as MHAKVCache with the fp16 head_dim axis replaced by the
packed block (66 B turbo4 / 130 B turbo8 / 52 B turbo3_tcq / 36 B turbo2_tcq
per 128-value group).

``store_kv`` quantizes the incoming fp16 K/V rows straight into the slab at
``out_loc`` rows (capture-safe: no syncs, no dynamic allocation).
``k_cache``/``v_cache`` keep the MHAKVCache contract of returning a dense
tensor view — for turbo pools these return the materialized fp16 scratch, so
attention backends that call them per layer must use
``materialize(layer_id, page_table, cache_seqlens)`` instead (see fa.py/fi.py
turbo branches).

``page_size`` must be 1: blocks are per-token, and the radix cache's page
arithmetic assumes one token per page. Rebuild (``ft ctl cache --kv N``)
reallocates the packed slabs in place, exactly like MHAKVCache.rebuild.
"""

from __future__ import annotations

from typing import Sequence

import torch

from freetoken.kernel.turbo_kv import CODEC_SPECS, block_bytes
from .base import BaseKVCachePool
from .mha_pool import MHAKVCache


class TurboKVCache(MHAKVCache):
    """MHAKVCache with packed quantized storage and a materializer."""

    # InnerQ calibration window (kv_codec_tune == "innerq"): the first
    # CALIBRATION_TOKENS stored tokens accumulate raw-domain per-channel
    # stats; then the pool computes buun-formula scales and uploads them.
    # One global [128] accumulator set shared by K and V (buun semantics).
    CALIBRATION_TOKENS = 2048

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        codec: str,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        if codec not in CODEC_SPECS:
            raise ValueError(f"unknown kv codec {codec!r}")
        if head_dim != 128:
            raise ValueError(
                f"turbo KV codecs require head_dim == 128 (rotation group), got {head_dim}"
            )
        if page_size != 1:
            raise ValueError("turbo KV pools require page_size=1")
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"turbo KV quantize needs fp16/bf16/fp32 input, got {dtype}")
        self._codec = codec
        self._bb = block_bytes(codec)
        self._input_dtype = dtype
        self._num_pages = num_pages
        self._page_size = page_size
        # The base class allocates fp16 slabs we never use — at 4097 pages x
        # 48 layers that's ~1 GiB wasted transiently. Allocate a 1-page stub to
        # satisfy the base's bookkeeping, then swap in the real packed slabs
        # (sized from THIS __init__'s num_pages, not the stub's).
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=1,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_packed()

    # ---- InnerQ calibration --------------------------------------------------------

    def arm_innerq_calibration(self) -> None:
        """Arm the calibration window: subsequent quantize calls accumulate
        raw-domain per-channel stats until CALIBRATION_TOKENS are stored."""
        from freetoken.kernel import turbo_kv
        from freetoken.utils import init_logger
        self._calib_tokens = 0
        self._calib_armed = True
        turbo_kv.innerq_arm_calibration(self._codec)
        init_logger(__name__).info(
            "InnerQ calibration armed (%s): accumulating raw K/V stats over "
            "the first %d stored tokens", self._codec, self.CALIBRATION_TOKENS)

    def _note_calibration_tokens(self, n_tokens: int) -> None:
        if not getattr(self, "_calib_armed", False):
            return
        from freetoken.kernel import turbo_kv
        from freetoken.utils import init_logger
        self._calib_tokens += n_tokens
        if self._calib_tokens < self.CALIBRATION_TOKENS:
            return
        self._calib_armed = False
        sq, ch_max, count = turbo_kv.innerq_download_stats(self._codec)
        res = turbo_kv.innerq_finalize_scales(sq, ch_max, count)
        log = init_logger(__name__)
        if res is None:
            # Identity upload disarms and keeps the codec exact: channels
            # already balanced (max ratio < 1.2) or the window saw no data.
            turbo_kv.innerq_upload_scales(
                self._codec, torch.ones(128), torch.ones(128))
            log.info("InnerQ: channels already balanced after %d groups — "
                     "scales left at identity", count)
            return
        scale, max_ratio = res
        turbo_kv.innerq_upload_scales(self._codec, scale, 1.0 / scale)
        top = (scale - 1.0).abs().argmax()
        log.info("InnerQ calibration done: %d groups, max scale ratio %.3f "
                 "(channel %d scale %.3f)", count, max_ratio,
                 int(top.item()), float(scale[top].item()))

    @property
    def innerq_calibrated(self) -> bool:
        """The calibration window ran to completion (scales uploaded, armed
        window closed). Identity-outcome pools also read True — the window
        ran, the codec just stayed exact (scales left at ones)."""
        return not getattr(self, "_calib_armed", False)

    # ---- storage -----------------------------------------------------------------

    def _alloc_packed(self) -> None:
        """Replace the inherited fp16 buffers with uint8 packed slabs."""
        _, num_storage_layers, _, _, local_kv_heads, _ = self._kv_buffer.shape
        self._kv_buffer = None
        self._k_buffer = None
        self._v_buffer = None
        shape = (
            num_storage_layers,
            self._num_pages * self._page_size,
            local_kv_heads,
            self._bb,
        )
        self._k_packed = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        self._v_packed = torch.zeros(shape, dtype=torch.uint8, device=self._device)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the packed slabs IN PLACE for ``num_pages`` pages.

        Packed bytes are codec-uniform across the pool, so a resize is a plain
        realloc (the radix cache drops prefixes on rebuild; the scheduler never
        re-reads old rows after a rejected-then-freed rebuild).
        """
        num_storage_layers, _old_tokens, local_kv_heads, _ = self._k_packed.shape
        self._num_pages = num_pages
        self._k_packed = None
        self._v_packed = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        shape = (num_storage_layers, num_pages * self._page_size, local_kv_heads, self._bb)
        self._k_packed = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        self._v_packed = torch.zeros(shape, dtype=torch.uint8, device=self._device)
        # Keep the base class's bookkeeping consistent (it rebuilds _kv_buffer;
        # ours is a stub — patch the fields it derives from _storage_shape).
        self._storage_shape = (num_pages * self._page_size, local_kv_heads, 128)

    # ---- write path ----------------------------------------------------------------

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import turbo_kv

        dense = self._dense(layer_id)
        heads = self._k_packed.shape[2]
        n_tokens = k.numel() // (heads * 128)
        self._note_calibration_tokens(n_tokens)
        # k/v arrive as strided slices of the fused qkv projection — quantize
        # needs contiguous rows.
        k3 = k.reshape(n_tokens, heads, 128).contiguous()
        v3 = v.reshape(n_tokens, heads, 128).contiguous()
        turbo_kv.turbo_quantize(
            self._codec, k3, self._k_packed[dense], out_loc, is_v=False,
        )
        turbo_kv.turbo_quantize(
            self._codec, v3, self._v_packed[dense], out_loc, is_v=True,
        )

    # ---- read path -------------------------------------------------------------------

    def materialize(
        self,
        layer_id: int,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the pages referenced by ``page_table`` into the scratch
        fp16 buffers and return ``(k, v)`` views shaped for the FA/FI backends:
        ``(bs, max_pages, heads, 128)`` fp16, page_table addressing intact.

        The dequant kernel writes output rows COMPACTED (input row i ->
        output row i), but the attention wrapper indexes the returned tensor
        by ORIGINAL page id (page_table values). Those diverge the moment a
        request's pages are not the pool's first ``n`` rows (prefix-cache
        reuse hands out freed page ids >= n) — the wrapper would read
        out-of-bounds scratch rows and attention sees garbage. So dequant
        into a staging buffer, then SCATTER to the page-id positions of the
        full-width scratch; the returned views are indexed by page id and
        cover the whole slab."""
        from freetoken.kernel import turbo_kv

        dense = self._dense(layer_id)
        # page_table arrives either 2-D (fa: bs x table_len) or a flattened 1-D
        # ragged index list (fi: concatenation of each request's device rows).
        flat = page_table.reshape(-1)
        heads = self._k_packed.shape[2]
        full = self._num_pages * self._page_size
        k_full = self._scratch(full, heads)
        v_full = self._scratch_v(full, heads)
        k_staging = torch.empty_like(k_full[: flat.numel()])
        v_staging = torch.empty_like(v_full[: flat.numel()])
        turbo_kv.turbo_dequantize(
            self._codec,
            self._k_packed[dense],
            k_staging,
            flat,
            is_v=False,
        )
        turbo_kv.turbo_dequantize(
            self._codec,
            self._v_packed[dense],
            v_staging,
            flat,
            is_v=True,
        )
        k_full[flat] = k_staging
        v_full[flat] = v_staging
        if page_table.dim() == 2:
            bs = page_table.shape[0]
            # full page-id-addressable width; callers index by page id
            return (
                k_full.view(bs, -1, heads, 128),
                v_full.view(bs, -1, heads, 128),
            )
        return k_full, v_full

    def _scratch(self, n: int, heads: int) -> torch.Tensor:
        """Persistent dequant scratch for K. Allocated ONCE at the full page-
        table width (max tokens) so a graph-captured reader can never gather a
        row beyond the buffer when the live batch is smaller than the plan's
        index-buffer size. Allocated in the pool's input dtype so backends
        consume it without a mid-graph cast (the dequant kernel emits
        fp16/bf16 to match)."""
        full = self._num_pages * self._page_size
        if not hasattr(self, "_k_scratch"):
            self._k_scratch = torch.empty(
                (full, heads, 128), dtype=self._input_dtype, device=self._device
            )
        return self._k_scratch[:n]

    def _scratch_v(self, n: int, heads: int) -> torch.Tensor:
        if not hasattr(self, "_v_scratch"):
            self._v_scratch = torch.empty(
                (self._num_pages * self._page_size, heads, 128),
                dtype=self._input_dtype, device=self._device,
            )
        return self._v_scratch[:n]

    # ---- cost model ---------------------------------------------------------------

    def unit_bytes(self) -> tuple[int, int]:
        # (heads * block_bytes) per token per K/V tensor + the dummy-page headroom
        # is folded into num_pages by the caller; report the true packed cost.
        tokens = self._k_packed.shape[1]
        per_token = (self._k_packed.numel() + self._v_packed.numel()) // max(tokens, 1)
        return per_token, 0

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        codec = getattr(config, "kv_codec", None) or "f16"
        if codec not in CODEC_SPECS:
            return super().kv_cost(config)
        bb = block_bytes(codec)
        # packed bytes per token per group: heads * head_dim/128 groups * bb,
        # times 2 (K and V). The dummy-page headroom follows the f16 convention.
        from .base import spec_kv_bytes_per_token

        # f16 bytes/token per group -> packed: divide by 2*head_dim/128 (f16
        # stores 2 B/elem x head_dim; packed stores bb B per 128-elem group per
        # slab) — i.e. multiply by bb / (2 * head_dim / 128). Keep the layer
        # and head-division terms from the f16 formula.
        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            if spec.head_dim != 128:
                # The pool rejects non-128 head dims at construction; price
                # at the f16 rate so an unsupported config never under-budgets.
                return super().kv_cost(config)
            f16 = spec_kv_bytes_per_token(spec, config)
            per_token += f16 * bb // (2 * spec.head_dim // 128 * 128)
        
        return per_token * config.page_size, 0, config.page_size, 0

    @property
    def dtype(self) -> torch.dtype:
        # The pool's INPUT dtype (what store_kv receives); the packed slabs are
        # uint8. Backends build scratch in this dtype.
        return self._input_dtype

    @property
    def is_turbo(self) -> bool:
        return True

    @property
    def codec(self) -> str:
        return self._codec