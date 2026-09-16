from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB

if TYPE_CHECKING:
    from freetoken.attention.triton import TritonAttentionBackend
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        # TurboQuant/TCQ pools: the materializer inside forward breaks
        # capture (illegal access under flashinfer's paged kernels). The
        # Triton backend's wave-2 fused decode reads the packed slabs
        # directly — decode no longer touches the materializer — so capture
        # is safe there. Other backends keep the eager behavior.
        kv_pool = getattr(self.attn_backend, "kvcache", None)
        from freetoken.attention.triton import TritonAttentionBackend

        if getattr(kv_pool, "is_turbo", False) and not isinstance(
            self.attn_backend, TritonAttentionBackend
        ):
            self.max_graph_bs = 0
            self.graph_bs_list = []
            return logger.info_rank0(
                "CUDA graph is disabled: --kv-codec turbo pool materialization "
                "is not capture-safe on this attention backend (triton enables it)."
            )
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        gc.collect()


def _determine_draft_graph_bs(
    max_running_req: int,
    cuda_graph_max_bs: int | None,
) -> List[int]:
    """The draft family's bs set: EVERY bs in 1..cap, no padding.

    The spec decode batch is one draft row per req, so it is bounded by
    max_running_req (4 on the 35B server) -- NOT by the trunk's
    throughput-motivated 160-bs family. Unlike the trunk, the draft's
    graphs must capture EVERY bs in the range (not a 1/2/4/8 ladder):
    the draft's MTPMoE bf16 bmm is batch-shape-sensitive -- a padded
    bs=4 kernel computes a live row's routed-expert sums with a different
    cuBLAS reduction order than the eager bs=3 kernel, and the draft
    logits/argmax diverge from the eager reference (measured: row-0
    carry row-sum -84.36 padded vs -90.44 exact at the same inputs).
    Padding is only bit-safe when the padded stage is bit-identical, and
    the draft MoE is not. Exact-bs graphs also remove the padded-tail
    machinery entirely: every replay runs the same shapes it was
    captured with, and each graphed step matches the eager path's
    kernel geometry exactly. Family size is <= max_running_req graphs.
    """
    cap = min(max_running_req, cuda_graph_max_bs if cuda_graph_max_bs else 160)
    if cuda_graph_max_bs is not None and cuda_graph_max_bs < 1:
        return []  # explicit 0 disables capture (the trunk's contract)
    return list(range(1, cap + 1))


def _draft_family_bs(graph_bs_list: List[int], bs: int) -> int | None:
    """The family's EXACT-bs entry for ``bs`` (None when bs is over the
    family -- the scheduler's eager-fallback predicate, kept a pure
    function so the CPU tests can pin it without CUDA). Exact-bs: a bs
    between family entries no longer pads; the family simply covers
    every bs in 1..cap, so only an over-cap bs falls back to eager."""
    return bs if bs in graph_bs_list else None


@dataclass
class MTPDraftBuffer:
    """Static I/O buffers of the captured MTP draft step (one per family,
    sized [max_bs, ...] and sliced [:padded_bs] per replay).

    Deliberately NOT the trunk GraphCaptureBuffer: the draft is 1 token
    per req with a [bs, H] carry input and no GDN state, so the shapes
    (and the invariants over them) differ. logits is fp32: the argmax runs
    INSIDE the captured graph over the fp32 copy of the bf16 lm_head
    output; bf16->fp32 is an exact cast, so the argmax (ties included --
    resolved index-wise over identical values) is bit-identical to the
    eager path's argmax over raw bf16 logits. carry_out is captured even
    though the scheduler discards it (the next iteration recomputes
    carries off the verify rows): the bit-equality gate compares it.
    """
    carry_in: torch.Tensor    # [max_bs, H] model dtype
    token_in: torch.Tensor    # [max_bs] int32
    out_loc: torch.Tensor     # [max_bs] int32 -- the layer-40 KV write slots
    positions: torch.Tensor   # [max_bs] int32
    logits: torch.Tensor      # [max_bs, vocab] fp32
    carry_out: torch.Tensor   # [max_bs, H] model dtype
    drafts: torch.Tensor      # [max_bs] int32

    @classmethod
    def init(cls, max_bs: int, hidden_size: int, vocab_size: int,
             dtype: torch.dtype, device: torch.device) -> MTPDraftBuffer:
        return cls(
            carry_in=torch.zeros(max_bs, hidden_size, dtype=dtype, device=device),
            token_in=torch.zeros(max_bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
            logits=torch.empty(max_bs, vocab_size, dtype=torch.float32, device=device),
            carry_out=torch.empty(max_bs, hidden_size, dtype=dtype, device=device),
            drafts=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )

    def stage_inputs(self, carries: torch.Tensor, tokens: torch.Tensor,
                     out_loc: torch.Tensor, positions: torch.Tensor,
                     bs: int) -> None:
        """Copy one replay step's live inputs into the static slices [:bs].
        The padded tail is NOT touched here -- the runner fills a benign
        tail separately, so tests can pin staging never clobbers it."""
        self.carry_in[:bs] = carries
        self.token_in[:bs] = tokens
        self.out_loc[:bs] = out_loc
        self.positions[:bs] = positions


class MTPDraftGraphRunner:
    """CUDA-graph family for the MTP draft step (wave-2 stage 2).

    Captures MTPHead.draft_step per bs (embed+norm -> fc -> the draft
    layer's paged attention + MTPMoE -> norm -> lm_head -> argmax) so the
    spec loop's per-iteration draft launch overhead disappears. The verify
    row forwards stay EAGER this stage, and the TRUNK GraphRunner stays
    disabled under --spec-mtp -- this family is the only capture.

    Mirrors GraphRunner: warmup forward OUTSIDE the graph (triton
    autotune / cublas handle setup must not be recorded), then capture on
    the engine stream with one shared mempool across the family. The draft
    batch's FI decode wrapper is the per-bs graph wrapper
    (attn_backend.prepare_for_capture); each replay re-plans the FI
    metadata host-side (prepare_for_replay) BEFORE g.replay() --
    flashinfer plan() is host work and must never be captured.

    The layer-40 KV row write rides the captured attention's out_loc read:
    out_loc is a static buffer the replay stages with the live slots, so
    every replay writes the right pool rows. The draft's GDN machinery does
    NOT apply (MTPDraftLayer has no GDN layers) -- no fla_metadata is built
    for the draft batch.

    EXACT-bs capture: the family holds one graph per bs in 1..cap (see
    _determine_draft_graph_bs). No padding, no dummy tail rows on replay
    -- the captured kernel geometry matches the eager path's per-bs call,
    which the bit-equality gate requires (the draft MoE's bf16 bmm is
    batch-shape-sensitive).
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        mtp,  # MTPHead
        attn_backend: BaseAttnBackend,
        max_running_req: int,
        cuda_graph_max_bs: int | None,
        max_seq_len: int,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        dummy_req: Req,
    ) -> None:
        bs_list = _determine_draft_graph_bs(max_running_req, cuda_graph_max_bs)
        self.attn_backend = attn_backend
        self.max_graph_bs = max(bs_list) if bs_list else 0
        self.graph_bs_list = sorted(bs_list)
        self.stream = stream
        self.device = device
        self.mtp = mtp
        self.dummy_req = dummy_req
        assert mtp is not None, "draft graph capture requires an attached MTP head"
        self._capture_graphs(max_seq_len, vocab_size, hidden_size, dtype)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int,
                        hidden_size: int, dtype: torch.dtype) -> None:
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            self.buffer = None
            return logger.info_rank0("MTP draft CUDA graph is disabled.")
        # The FI backend's capture state is free here: the trunk GraphRunner
        # stays disabled under --spec-mtp, so it never called
        # init_capture_graph and graph_wrappers is empty.
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len,
                                             bs_list=self.graph_bs_list)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        logger.info_rank0(f"Capturing MTP draft graphs with sizes: {self.graph_bs_list}")
        self.buffer = MTPDraftBuffer.init(
            self.max_graph_bs, hidden_size, vocab_size, dtype, self.device)
        # The scratch slot the CAPTURE-time KV writes land in (the engine
        # fills page_table[dummy] with this slot id at init). Parked
        # BEFORE the warmup so capture-time store_kv touches scratch, not
        # live pool row 0. Exact-bs replays never use this: every staged
        # row is a real row with real out_locs.
        self._dummy_out_loc = int(
            get_global_ctx().page_table[self.dummy_req.table_idx, 0].item())
        self.buffer.out_loc.fill_(self._dummy_out_loc)
        mtp = self.mtp
        pool = None
        for bs in sorted(self.graph_bs_list, reverse=True):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            # Alias the static buffers onto the batch: the captured kernels
            # record THESE addresses, and every replay stages live values
            # into them (input_ids rides the ctx batch only for the warmup;
            # the captured embed reads token_in).
            batch.input_ids = self.buffer.token_in[:bs]
            batch.out_loc = self.buffer.out_loc[:bs]
            batch.positions = self.buffer.positions[:bs]
            with get_global_ctx().forward_batch(batch):
                _, warm_logits = mtp.draft_step(
                    self.buffer.carry_in[:bs], self.buffer.token_in[:bs])
                self.buffer.logits[:bs] = warm_logits
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    carry, logits = mtp.draft_step(
                        self.buffer.carry_in[:bs], self.buffer.token_in[:bs])
                    self.buffer.carry_out[:bs] = carry
                    # fp32 stage + argmax INSIDE the graph: an exact cast,
                    # so the argmax (ties included) is bit-identical to the
                    # eager path's argmax over the raw bf16 logits.
                    self.buffer.logits[:bs] = logits
                    self.buffer.drafts[:bs] = self.buffer.logits[:bs].argmax(
                        dim=-1).to(torch.int32)
            if pool is None:
                pool = graph.pool()  # one shared mempool across the family
            self.graph_map[bs] = graph

    def can_draft(self, bs: int) -> bool:
        """The scheduler's eager-fallback gate: False when capture is
        disabled (FT_SPEC_DRAFT_EAGER leaves the runner None) or bs has
        no exact family entry (bs over the family cap)."""
        return self.max_graph_bs > 0 and _draft_family_bs(self.graph_bs_list, bs) is not None

    def draft(self, carries: torch.Tensor, tokens: torch.Tensor,
              draft_batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """One replay step: (carries [bs, H], tokens [bs] int32) ->
        (drafts [bs] int32, carry' [bs, H]). EXACT-bs: the replay runs the
        graph captured at this very bs -- no padding, no discarded tail
        rows, kernel geometry identical to the eager path's.

        ``draft_batch`` is the scheduler's row-A spec batch
        (_make_spec_row_batch(is_row_a=True)): it supplies the req views
        whose device_len/table_idx feed the FI metadata plan and whose
        positions/out_loc this method stages into the static buffers. A
        fresh FIMetadata is built EVERY step (prepare_for_replay asserts
        uninitialized -- plan() is host work outside the captured region).
        """
        bs = draft_batch.size
        assert bs == carries.shape[0] == tokens.shape[0], (
            f"draft replay shape mismatch: bs={bs} carries={carries.shape} tokens={tokens.shape}")
        assert _draft_family_bs(self.graph_bs_list, bs) is not None, (
            f"bs={bs} over the draft family {self.graph_bs_list}: "
            "the scheduler must gate through can_draft() first")
        assert self.buffer is not None and self.max_graph_bs > 0
        g = self.graph_map[bs]
        self.attn_backend.prepare_metadata(draft_batch)
        self.buffer.stage_inputs(
            carries, tokens, draft_batch.out_loc, draft_batch.positions, bs)
        self.attn_backend.prepare_for_replay(draft_batch)
        g.replay()
        return self.buffer.drafts[:bs], self.buffer.carry_out[:bs]

    # NOTE: must run before freeing NCCL resources (same contract as the trunk).
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND
        # the static buffer tensors; dropping the references is the
        # load-bearing step for the rebuild's free-before-alloc.
        self.graph_map = {}
        self.buffer = None
        gc.collect()


@dataclass
class MTPVerifyBuffer:
    """Static I/O buffers of the captured verify-row forward (one family,
    sized [max_bs, ...], sliced [:bs] per replay).

    Both verify rows have IDENTICAL captured geometry (bs rows x 1 token,
    decode phase) and differ only in staged buffer contents, so ONE buffer
    + ONE graph per bs serves row A and row B. hidden_out is the captured
    post-trunk hidden (the next row's B-input carry + the resolve's
    carry); logits is fp32 (exact bf16 cast — the argmax comparisons are
    bit-identical to the eager path's). No GDN snapshot state here: the
    mid-verify snapshot stays a host-issued pool copy BETWEEN the two
    replays (program order = stream order, the same invariant the eager
    path relies on).
    """
    token_in: torch.Tensor    # [max_bs] int32 — row A: spec_next_input, row B: the draft
    out_loc: torch.Tensor     # [max_bs] int32 — the row's KV write slot
    positions: torch.Tensor   # [max_bs] int32
    logits: torch.Tensor      # [max_bs, vocab] fp32
    hidden_out: torch.Tensor  # [max_bs, H] model dtype — post-trunk last_hidden

    @classmethod
    def init(cls, max_bs: int, hidden_size: int, vocab_size: int,
             dtype: torch.dtype, device: torch.device) -> MTPVerifyBuffer:
        return cls(
            token_in=torch.zeros(max_bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
            positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
            logits=torch.empty(max_bs, vocab_size, dtype=torch.float32, device=device),
            hidden_out=torch.empty(max_bs, hidden_size, dtype=dtype, device=device),
        )

    def stage_inputs(self, tokens: torch.Tensor, out_loc: torch.Tensor,
                     positions: torch.Tensor, bs: int) -> None:
        """Copy one row's live inputs into the static slices [:bs]."""
        self.token_in[:bs] = tokens
        self.out_loc[:bs] = out_loc
        self.positions[:bs] = positions


class MTPVerifyGraphRunner:
    """CUDA-graph family for the MTP verify trunk row (wave-2 stage 3).

    One captured TRUNK forward per bs (the full decoder stack + lm_head +
    last_hidden store), replayed twice per spec iteration: row A
    (the certain token at position q) and row B (the draft at q+1). The
    two rows have identical captured geometry (bs rows x 1 token, decode
    phase) — only the staged buffers and the FI plan (host-side, outside
    the graph) differ — so one graph per bs covers both rows.

    Mirrors MTPDraftGraphRunner: EXACT-bs family (no padding; the verify
    MoE's bf16 GEMM is M-shape-sensitive), warmup outside the graph, one
    shared mempool, FI decode wrapper per bs via
    attn_backend.prepare_for_capture; each replay re-plans the FI
    metadata host-side (prepare_for_replay) BEFORE g.replay().

    The mid-verify GDN snapshot (defect-2 fix) stays a host-issued
    linear_state_pool copy issued BETWEEN the two replays — program order
    on the engine stream puts it after row A's kernels and before row
    B's, exactly matching the eager path. The GDN state pool slots are
    keyed by linear_table_idx, which the captured GDN kernels read from
    static buffers staged per replay (same mechanism as the draft's
    out_loc).
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model,  # Qwen3_5MoEForCausalLM (the TRUNK)
        attn_backend: BaseAttnBackend,
        max_running_req: int,
        cuda_graph_max_bs: int | None,
        max_seq_len: int,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype,
        dummy_req: Req,
        moe_offload_cache: "OffloadMoeCache | None" = None,
    ) -> None:
        bs_list = _determine_draft_graph_bs(max_running_req, cuda_graph_max_bs)
        self.attn_backend = attn_backend
        self.max_graph_bs = max(bs_list) if bs_list else 0
        self.graph_bs_list = sorted(bs_list)
        self.stream = stream
        self.device = device
        self.model = model
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.table_bufs: Dict[int, torch.Tensor] = {}
        self._capture_graphs(max_seq_len, vocab_size, hidden_size, dtype)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int,
                        hidden_size: int, dtype: torch.dtype) -> None:
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            self.buffer = None
            return logger.info_rank0("MTP verify CUDA graph is disabled.")
        # The FI backend's capture state is ALREADY ARMED here: the trunk
        # GraphRunner stays disabled under --spec-mtp and the DRAFT runner
        # (constructed before this one — engine.py ordering contract) has
        # called init_capture_graph and built its per-bs graph_wrappers.
        # The verify runner reuses that same arm: prepare_for_capture below
        # adds the verify's own wrappers (per-bs, keyed the same way) —
        # both families' wrappers live in attn_backend.graph_wrappers
        # until reset_capture tears them down together.
        logger.info_rank0(f"Capturing MTP verify graphs with sizes: {self.graph_bs_list}")
        self.buffer = MTPVerifyBuffer.init(
            self.max_graph_bs, hidden_size, vocab_size, dtype, self.device)
        self._dummy_out_loc = int(
            get_global_ctx().page_table[self.dummy_req.table_idx, 0].item())
        self.buffer.out_loc.fill_(self._dummy_out_loc)
        # Persistent GDN cu_seqlens (one arange, sliced [:bs+1] per captured bs).
        # The captured GDN kernels load bos/eos from this memory AT REPLAY, so it
        # must have an owner that outlives the capture loop: the loop-local
        # capture Batch (whose fla_metadata holds the tensor) is dropped at the
        # next iteration, and a fresh per-bs arange here was freed and reused by
        # replay-time eager temps — corrupting bos/eos/T inside every captured
        # GDN kernel at bs>=2 (bs=1 stayed bit-equal only by allocator luck).
        self.fla_cu_seqlens = torch.arange(
            self.max_graph_bs + 1, dtype=torch.int32, device=self.device)
        pool = None
        for bs in sorted(self.graph_bs_list, reverse=True):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            # GDN metadata against static buffers (stable addresses): the
            # captured GDN kernels read cu_seqlens (constant arange) and
            # cache_indices (this bs's persistent slot map) from HERE.
            table_buf = torch.zeros(
                max(self.graph_bs_list), dtype=torch.int32, device=self.device)
            self.table_bufs[bs] = table_buf
            if self.linear_state_pool() is not None:
                from freetoken.attention.linear import FLAMetadata
                batch.fla_metadata = FLAMetadata(
                    cu_seqlens=self.fla_cu_seqlens[: bs + 1],
                    cache_indices=table_buf[:bs],
                )
            batch.input_ids = self.buffer.token_in[:bs]
            batch.out_loc = self.buffer.out_loc[:bs]
            batch.positions = self.buffer.positions[:bs]
            with get_global_ctx().forward_batch(batch):
                warm_logits = self.model.forward()
                self.buffer.logits[:bs] = warm_logits
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    logits = self.model.forward()
                    self.buffer.logits[:bs] = logits
                    self.buffer.hidden_out[:bs] = self.model.last_hidden.to(dtype)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # one shared mempool across the family
            self.graph_map[bs] = graph
        self._reset_moe_offload_cache()

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def linear_state_pool(self):
        return getattr(get_global_ctx(), "linear_state_pool", None)

    def can_verify(self, bs: int) -> bool:
        """The scheduler's eager-fallback gate (mirror of can_draft)."""
        return self.max_graph_bs > 0 and _draft_family_bs(self.graph_bs_list, bs) is not None

    def verify_row(self, row_batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """One replay: (tokens [bs] int32, out_loc [bs] int32, positions
        [bs] int32, linear slots [bs] int32 staged from ``row_batch``) ->
        (row_logits fp32 [bs, vocab], row_hidden [bs, H]).

        ``row_batch`` is the scheduler's 1-row spec row batch
        (_make_spec_row_batch): it supplies the req views whose
        device_len/table_idx feed the FI plan and whose
        positions/out_loc/linear_table_idx this method stages.
        """
        bs = row_batch.size
        assert _draft_family_bs(self.graph_bs_list, bs) is not None, (
            f"bs={bs} over the verify family {self.graph_bs_list}: "
            "the scheduler must gate through can_verify() first")
        assert self.buffer is not None and self.max_graph_bs > 0
        g = self.graph_map[bs]
        self.attn_backend.prepare_metadata(row_batch)
        self.buffer.stage_inputs(
            row_batch.input_ids, row_batch.out_loc, row_batch.positions, bs)
        if self.linear_state_pool() is not None:
            self.graph_map_table(row_batch, bs)
        self.attn_backend.prepare_for_replay(row_batch)
        g.replay()
        return self.buffer.logits[:bs], self.buffer.hidden_out[:bs]

    def graph_map_table(self, row_batch: Batch, bs: int) -> None:
        """Stage the row's GDN slot ids into the captured cache_indices.
        One persistent buffer per captured bs holds the row's
        linear_table_idx; the FLAMetadata built at capture points HERE."""
        slots = row_batch.linear_table_idx
        assert slots is not None, "verify replay requires linear slots (hybrid GDN)"
        buf = self.table_bufs[bs]
        buf[:bs] = slots.to(torch.int32, non_blocking=True)

    # NOTE: must run before freeing NCCL resources (same contract as the trunk).
    def destroy_cuda_graphs(self) -> None:
        self.graph_map = {}
        self.buffer = None
        self.fla_cu_seqlens = None
        gc.collect()
